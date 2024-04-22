import os
import dlib
import cv2
import numpy as np
from scipy.spatial import distance
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    company_collection = mongo.db[str(company_id)]
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

    db = mongo.db
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

app = Flask(__name__)
CORS(app)

app.config['MONGO_URI'] = 'mongodb+srv://alpha:alpha@cluster1.elvzeic.mongodb.net/Company'
mongo = PyMongo(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/post-face", methods=['POST'])
def post_face():
    try:
        company_id = request.form['company_id']
        user_id = request.form['user_id']
        username = request.form['username']

        existing_collections = mongo.db.list_collection_names()
        if str(company_id) not in existing_collections:
            # Create a new collection if the company ID does not exist
            mongo.db.create_collection(str(company_id))

        if 'File1' not in request.files or 'File2' not in request.files or 'File3' not in request.files:
            return jsonify({'error': 'Please upload all three image files.'}), 400

        file1 = request.files['File1']
        file2 = request.files['File2']
        file3 = request.files['File3']

        if company_id and user_id and username and file1 and file2 and file3:
            file_paths = []
            for file in [file1, file2, file3]:
                if file.filename == '':
                    return jsonify({'error': 'One of the files has no selected file.'}), 400
                if file:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(file_path)
                    file_paths.append(file_path)

            upload_labeled_images(file_paths, company_id, user_id, username)
            return jsonify({'message': 'Face model updated successfully'})
        else:
            return jsonify({'error': 'Incomplete form data provided.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/check-face", methods=['POST'])
def check_face():
    try:
        company_id = request.form['company_id']

        existing_collections = mongo.db.list_collection_names()
        if str(company_id) not in existing_collections:
            return jsonify({'error': f'Company model for {company_id} is not available. Please choose a company from {existing_collections}'}), 400

        if 'File1' not in request.files:
            return jsonify({'error': 'Please upload an image file to check.'}), 400

        file1 = request.files['File1']

        if file1.filename == '':
            return jsonify({'error': 'No selected file.'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(file_path)

        result = compare_faces(file_path, company_id)

        if isinstance(result, str):
            return jsonify({'result': result})
        elif len(result) == 0:
            return jsonify({'result': 'Face not recognized.'})
        elif len(result) > 1:
            return jsonify({'result': 'Multiple faces found'})
        else:
            return jsonify({'result': result[0]})
    except Exception as e:
        print("Check face error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
