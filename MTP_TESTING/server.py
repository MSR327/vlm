import socket

HOST = ""
PORT = 5000

with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
    s.bind((HOST,PORT))
    s.listen()
    print(f"Listening on port {PORT}....")
    conn , addr = s.accept()
    with conn:
        print("Connected by", addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print("Received: ",data.decode())
            conn.sendall(data)

