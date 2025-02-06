import serial
import time
import base64
import os

# Serial port configuration for the LoRa receiver
serial_port = '/dev/cu.usbserial-140'  # Update this to match your LoRa receiver's serial port
baud_rate = 115200
buffer = ""
csv_file = "received_data.csv"

# Ensure directory for received images exists
received_images_dir = "received_images"
os.makedirs(received_images_dir, exist_ok=True)  # Create folder if not exists

try:
    # Open the serial port
    serial_obj = serial.Serial(serial_port, baud_rate)
    print("✅ Serial port opened on receiver")
    print("🔹 Waiting for incoming message...")

    image_name = None  # Store the current image name
    mainmess = ""  # Store image data

    while True:
        # Check if there is incoming data
        if serial_obj.in_waiting > 0:
            chunk = serial_obj.read(serial_obj.in_waiting).decode('utf-8', errors='ignore')
            buffer += chunk  # Add chunk to buffer

            if '.?' in buffer:
                message, buffer = buffer.split('.?', 1)
                message = message.strip()
                print("📥 Received:", message)

                if message.startswith("csv,"):
                    # Save CSV data
                    with open(csv_file, "a") as f:
                        f.write(message[4:] + "\n")  # Skip "csv," and write the rest of the row
                        print(f"✅ CSV row saved: {message[4:]}")

                elif message == "image_name":
                    print("📥 Acknowledged: image_name (waiting for actual name)")

                    # Ensure the previous image (if any) is processed before setting a new one
                    if image_name and mainmess:
                        try:
                            decoded_image = base64.b64decode(mainmess)
                            output_file = os.path.join(received_images_dir, image_name)
                            with open(output_file, "wb") as image_file:
                                image_file.write(decoded_image)
                            print(f"✅ Image saved: {output_file}")
                        except Exception as e:
                            print(f"❌ Error decoding image: {e}")

                        mainmess = ""  # Reset for new image

                    # Wait for actual image name
                    start_time = time.time()
                    while (time.time() - start_time) < 3:  # Wait up to 3 seconds for filename
                        if serial_obj.in_waiting > 0:
                            filename_chunk = serial_obj.read(serial_obj.in_waiting).decode('utf-8', errors='ignore')
                            buffer += filename_chunk
                            if '.?' in buffer:
                                image_name, buffer = buffer.split('.?', 1)
                                image_name = image_name.strip()
                                print(f"🖼 Receiving image: {image_name}")
                                break  # Exit loop once filename is received
                        time.sleep(0.1)

                    if not image_name:
                        print("❌ Warning: No image name received! Skipping image.")
                        continue  # Skip to next loop iteration if no filename is received

                elif message.endswith(".jpg") and len(message) < 50:
                    # If a new image name is received, save the previous image first
                    if image_name and mainmess:
                        try:
                            decoded_image = base64.b64decode(mainmess)
                            output_file = os.path.join(received_images_dir, image_name)
                            with open(output_file, "wb") as image_file:
                                image_file.write(decoded_image)
                            print(f"✅ Image saved: {output_file}")
                        except Exception as e:
                            print(f"❌ Error decoding image: {e}")

                        mainmess = ""  # Reset for new image

                    image_name = message  # Store new image filename
                    print(f"🖼 Receiving image: {image_name}")

                else:
                    # Accumulate image data
                    mainmess += message

                buffer = ""

            time.sleep(0.1)  # Avoid busy waiting

except serial.SerialException as e:
    print(f"❌ Error with serial connection: {e}")

finally:
    # Save the last image (if any) before exiting
    if image_name and mainmess:
        try:
            decoded_image = base64.b64decode(mainmess)
            output_file = os.path.join(received_images_dir, image_name)
            with open(output_file, "wb") as image_file:
                image_file.write(decoded_image)
            print(f"✅ Final image saved: {output_file}")
        except Exception as e:
            print(f"❌ Error decoding final image: {e}")

    if 'serial_obj' in locals() and serial_obj.is_open:
        serial_obj.close()
        print("✅ Serial port closed on receiver")

