import cv2
import numpy as np
import tensorflow.lite as tflite

# **Load TFLite Model**
interpreter = tflite.Interpreter(model_path="drowsiness_model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (128, 128)

def predict_drowsiness(image):
    """ Preprocess image and run inference """
    img = cv2.resize(image, IMG_SIZE) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0).astype(np.float32)  # Expand dimensions

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    return "Drowsy" if output[0][0] > 0.5 else "Not Drowsy"

# **Open Webcam and Detect Drowsiness**
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get prediction
    label = predict_drowsiness(frame)

    # Display result
    cv2.putText(frame, f"Status: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
