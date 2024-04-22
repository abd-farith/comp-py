import os
import dlib
import cv2
import numpy as np
from scipy.spatial import distance
import streamlit as st
from pymongo import MongoClient

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_face_encodings(image_path):
    face_encodings = []
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    # Load the face recognition model
    face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    for face in faces:
        shape = shape_predictor(image, face)
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        face_encodings.append(np.array(face_descriptor))

    return face_encodings

def compare_faces(image_path, company_id):
    results = []
    face_encodings = extract_face_encodings(image_path)

    if not face_encodings:
        return "No face detected in the image"

    if len(face_encodings) > 1:
        return "Multiple faces detected. Cannot process the image"

    query_encoding = face_encodings[0]

    # MongoDB connection and query logic
    client = MongoClient("mongodb+srv://alpha:alpha@cluster1.elvzeic.mongodb.net/Company")
    db = client.Company
    company_collection = db[str(company_id)]

    for face in company_collection.find():
        for descriptor in face['descriptions']:
            distance_score = distance.euclidean(query_encoding, descriptor)
            if distance_score < 0.4:
                results.append({'user_id': face['user_id'], 'username': face['username']})
                break

    return results

def upload_labeled_images(images, company_id, user_id, username):
    descriptions = []
    for img_path in images:
        face_encodings = extract_face_encodings(img_path)
        if len(face_encodings) == 1:
            descriptions.append(face_encodings[0].tolist())

    # MongoDB connection and update logic
    client = MongoClient("mongodb+srv://alpha:alpha@cluster1.elvzeic.mongodb.net/Company")
    db = client.Company
    company_collection = db[str(company_id)]
    existing_user = company_collection.find_one({'user_id': user_id})

    if existing_user:
        company_collection.update_one(
            {'user_id': user_id},
            {'$set': {'username': username, 'descriptions': descriptions}}
        )
    else:
        company_collection.insert_one({
            'user_id': user_id,
            'username': username,
            'descriptions': descriptions
        })

# Streamlit app starts here

st.title("Face Recognition System")

option = st.sidebar.selectbox("Select Action", ["Check Face", "Post Face"])

if option == "Check Face":
    st.subheader("Check Face")
    company_id = st.text_input("Enter Company ID:")
    file_to_check = st.file_uploader("Upload an image to check:", type=["jpg", "jpeg", "png"])

    if st.button("Check"):
        if file_to_check is not None:
            file_path = os.path.join("uploads", file_to_check.name)
            with open(file_path, "wb") as f:
                f.write(file_to_check.read())
            result = compare_faces(file_path, company_id)
            st.write(result)
        else:
            st.error("Please upload an image to check.")

elif option == "Post Face":
    st.subheader("Post Face")
    company_id = st.text_input("Enter Company ID:")
    user_id = st.text_input("Enter User ID:")
    username = st.text_input("Enter Username:")
    uploaded_files = st.file_uploader("Upload three images:", accept_multiple_files=True)

    if st.button("Upload"):
        if uploaded_files is not None and len(uploaded_files) == 3:
            file_paths = []
            for file in uploaded_files:
                file_path = os.path.join("uploads", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                file_paths.append(file_path)
            upload_labeled_images(file_paths, company_id, user_id, username)
            st.success("Images uploaded successfully.")
        else:
            st.error("Please upload three images.")
