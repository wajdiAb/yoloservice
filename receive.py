import pika

# Create a connection to RabbitMQ server
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

queue_name = 'hello'
channel.queue_declare(queue=queue_name)

# Define a callback function to process received messages
def callback(ch, method, properties, body):
    print(f" [x] Received {body.decode()}")

# Set up consumer
channel.basic_consume(queue=queue_name,
                     auto_ack=True,  # Automatic acknowledgment mode
                     on_message_callback=callback)

print(' [*] Waiting for messages. To exit press CTRL+C')

# Start consuming
channel.start_consuming()