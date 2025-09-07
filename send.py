import pika

# Create a connection to RabbitMQ server
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()


queue_name = 'hello'
channel.queue_declare(queue=queue_name)

message = 'Hello World!'
channel.basic_publish(exchange='',
                     routing_key=queue_name,
                     body=message)
print(f" [x] Sent '{message}'")

connection.close()