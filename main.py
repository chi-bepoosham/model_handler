import pika
import os
import json
import re
import time
import requests
import random
import logging
from model_handler_service.load_and_predict_man import process_clothing_image, get_man_body_type
from model_handler_service.load_and_predict_woman import process_woman_clothing_image, process_six_model_predictions, get_body_type_female
from model_handler_service.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_image(gender,action,image_link):
    # Get temp directory from config
    temp_images_dir = config.get_temp_dir()

    filename = image_link.split('/')[-1]
    img_data = requests.get(image_link).content
    img_name = str(temp_images_dir / f"{int(time.time())}{random.randrange(100, 999)}-temp-{filename}")

    with open(img_name, 'wb') as handler:
        handler.write(img_data)

    if gender == 1 or gender == '1':
        if action == 'body_type':
            process_data = get_man_body_type(img_name)
        else:
            process_data = process_clothing_image(img_name)
    else:
        if action == 'body_type':
            process_data = get_body_type_female(img_name)
        else:
            process_data = process_woman_clothing_image(img_name)
            if process_data.get('paintane') is None:
                process_data = process_six_model_predictions(img_name)

    return {
        "process_data":process_data,
        "item_name":"image_processed",
        "item_content":"1d5w1dw4d6w4d6w46d"
    }

def establish_connection():
    # Get RabbitMQ config from config object
    rabbitmq_config = config.get_rabbitmq_config()
    
    logger.info(f"Attempting to connect to RabbitMQ at {rabbitmq_config['host']}...")
    
    credentials = pika.PlainCredentials(
        rabbitmq_config['user'],
        rabbitmq_config['password']
    )
    parameters = pika.ConnectionParameters(
        host=rabbitmq_config['host'],
        virtual_host=rabbitmq_config['vhost'],
        credentials=credentials
    )

    connection = pika.BlockingConnection(parameters)
    logger.info("Successfully connected to RabbitMQ!")
    return connection

def process_message(ch, method, properties, body):
    if properties.headers and properties.headers.get('index') == 1:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    else:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        message = json.loads(body)
        job_id = message.get("id")
        job_uuid = message.get("uuid")
        job_name = message.get("displayName")

        if job_name == 'App\\Jobs\\SendRabbitMQMessage':
            try:
                try:
                    php_serialized_data = message.get("data", {}).get("command")
                    match = re.search(r's:7:"\x00\*\x00data";a:\d+:{(.*)}', php_serialized_data)
                    if match:
                        data_content = match.group(1)
                        key_value_pairs = re.findall(r's:\d+:"(.*?)";(s|i|N):(?:\d+:"(.*?)"|([^;]*));?', data_content)

                        messageData = {}
                        for key, type_, str_value, int_or_null in key_value_pairs:
                            if type_ == 'i':
                                try:
                                    messageData[key] = int(int_or_null)
                                except ValueError:
                                    messageData[key] = None
                            elif type_ == 's':
                                messageData[key] = str_value
                            elif type_ == 'N':
                                messageData[key] = None
                    else:
                        messageData = None

                except TypeError as e:
                    messageData = None

            except (json.JSONDecodeError, KeyError) as e:
                messageData = None

            if messageData is not None:
                action = messageData.get("action")
                user_id = messageData.get("user_id")
                gender = messageData.get("gender")
                clothes_id = messageData.get("clothes_id")
                image_link = messageData.get("image_link")
                time = messageData.get("time")
                process_image_data = process_image(gender,action,image_link)

                completion_data = {
                    "id": job_id,
                    "uuid": job_uuid,
                    "job": job_name,
                    "data": {
                        "process_image": process_image_data,
                        "action": action,
                        "user_id": user_id,
                        "gender": gender,
                        "clothes_id": clothes_id,
                        "image_link": image_link,
                        "time": time,
                    },
                }

                send_message_to_rabbitmq(completion_data)

def send_message_to_rabbitmq(data):
    """Send a completion message to the RabbitMQ response queue."""
    connection = establish_connection()
    channel = connection.channel()

    message = json.dumps(data)
    channel.basic_publish(
        exchange='',
        routing_key=config.get_rabbitmq_config()['response_queue'],
        body=message,
        properties=pika.BasicProperties(delivery_mode=2,headers={'index': 1})
    )

    connection.close()

def consume_queue():
    logger.info("Starting the model handler service...")
    time.sleep(10)
    
    try:
        connection = establish_connection()
        channel = connection.channel()

        response_queue = config.get_rabbitmq_config()['response_queue']
        channel.queue_declare(queue=response_queue, durable=True, passive=True)
        channel.basic_consume(queue=response_queue, on_message_callback=process_message)
        
        logger.info(f"Waiting for messages on queue '{response_queue}'. To exit press CTRL+C")
        channel.start_consuming()
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
        raise
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
        if connection and not connection.is_closed:
            connection.close()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    consume_queue()
