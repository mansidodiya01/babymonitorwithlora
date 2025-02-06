import serial
import time
import base64
import os
import csv

# Serial port configuration
serial_port = '/dev/cu.usbserial-130'  # Change as needed
baud_rate = 115200
image_folder = "images/cropped"  # Folder containing images to send
sent_images_file = "sent_images.txt"
csv_file = "baby_monitoring_log.csv"  # CSV file containing metadata
chunk_size = 198  # Define chunk size

def load_sent_images():
    """Load names of already sent images to avoid resending."""
    if os.path.exists(sent_images_file):
        with open(sent_images_file, "r") as f:
            return set(f.read().splitlines())  # Store sent images in a set
    return set()

def mark_image_as_sent(image_name):
    """Mark an image as sent in the log file."""
    with open(sent_images_file, "a") as f:
        f.write(image_name + "\n")  # Append image name to the file

def get_csv_rows():
    """Load all rows from the CSV, ensuring all metadata is sent."""
    if not os.path.exists(csv_file):
        print(f"⚠️ CSV file '{csv_file}' not found.")
        return []

    rows = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            rows.append(row)
    return rows

try:
    # Open the serial port
    serial_obj = serial.Serial(serial_port, baud_rate)
    time.sleep(2)  # Allow time for initialization
    print("✅ Serial port opened on sender")

    csv_rows = get_csv_rows()  # Get all CSV rows
    sent_images = load_sent_images()  # Load previously sent images

    if not csv_rows:
        print("✅ No CSV rows to send. Exiting.")
    else:
        for row in csv_rows:
            timestamp, cry_status, detect_status, image_name = row  # Unpack CSV row

            # Send CSV data
            csv_message = "csv," + ",".join(row) + ".?"  # Prefix "csv," for identification
            serial_obj.write(csv_message.encode('utf-8'))
            print(f"📤 Sent CSV row: {csv_message}")
            time.sleep(2)  # Allow time before potentially sending an image

            # **Check if image should be sent**
            if image_name.strip().lower() == "none" or image_name in sent_images:
                print(f"⚠️ Skipping image transmission for '{image_name}' (Metadata-only mode).")
                continue  # Skip sending the image if it's marked "None"

            # **Check if image exists before sending**
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                print(f"⚠️ Image '{image_name}' not found. Sending metadata only.")
                continue  # Skip sending the image if it does not exist

            # Send image name
            serial_obj.write("image_name.?".encode('utf-8'))
            image_name_msg = image_name + ".?"
            serial_obj.write(image_name_msg.encode('utf-8'))
            print(f"📤 Sent image name: {image_name}")
            time.sleep(4)

            # Read and send image data
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            # Encode image data as Base64 and send in chunks
            message = base64.b64encode(image_data).decode('utf-8')
            for i in range(0, len(message), chunk_size):
                chunk = message[i:min(i + chunk_size, len(message))] + ".?"
                serial_obj.write(chunk.encode('utf-8'))
                print(f"📤 Sent chunk: {chunk}")
                time.sleep(3)  # Adjust delay as needed

            mark_image_as_sent(image_name)  # Mark the image as sent
            print(f"✅ Marked {image_name} as sent.")

    print("✅ Finished sending CSV rows and images. Exiting.")

except serial.SerialException as e:
    print(f"❌ Error with serial connection: {e}")

finally:
    if 'serial_obj' in locals() and serial_obj.is_open:
        serial_obj.close()
        print("✅ Serial port closed on sender")
