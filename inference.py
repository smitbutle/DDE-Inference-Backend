import os
import csv
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification
from utils import unnormalize_box, extract_text_from_boxes, extract_label_word_pairs, draw_predictions_on_image
import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate("./documentdataextractor-16ad89fe2724.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'gs://documentdataextractor.appspot.com'})


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name} in {bucket_name}.")




processor_local_dir = "./layoutlmv3-processor"
model_local_dir = "./document-data-extraction-layoutlmv3-model"


if not os.path.exists(processor_local_dir):
    print("Downloading processor...")
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    processor.tokenizer.do_lower_case = False
    processor.save_pretrained(processor_local_dir)
else:
    print("Loading processor from local directory...")
    processor = AutoProcessor.from_pretrained(processor_local_dir)
    processor.tokenizer.do_lower_case = False
if not os.path.exists(model_local_dir):
    print("Downloading model...")
    model = AutoModelForTokenClassification.from_pretrained("smitbutle/document-data-extraction-layoutlmv3")
    model.save_pretrained(model_local_dir)
else:
    print("Loading model from local directory...")
    model = AutoModelForTokenClassification.from_pretrained(model_local_dir)


local_processor = AutoProcessor.from_pretrained(processor_local_dir)
local_processor.tokenizer.do_lower_case = False
local_model = AutoModelForTokenClassification.from_pretrained(model_local_dir)

labels =['O', 'B-ABN', 'B-BILLER', 'B-BILLER_ADDRESS', 'B-BILLER_POST_CODE', 'B-DUE_DATE', 'B-GST', 'B-INVOICE_DATE', 'B-INVOICE_NUMBER', 'B-SUBTOTAL', 'B-TOTAL', 'I-BILLER_ADDRESS']
id2label = {v: k for v, k in enumerate(labels)}

label2color = {
    "B-ABN": 'blue',
    "B-BILLER": 'blue',
    "B-BILLER_ADDRESS": 'green',
    "B-BILLER_POST_CODE": 'orange',
    "B-DUE_DATE": "blue",
    "B-GST": 'green',
    "B-INVOICE_DATE": 'violet',
    "B-INVOICE_NUMBER": 'orange',
    "B-SUBTOTAL": 'green',
    "B-TOTAL": 'blue',
    "I-BILLER_ADDRESS": 'blue',
    "O": 'orange'
}

def write_to_csv(data, keys):
    file_exists = os.path.isfile('data.csv')

    with open('data.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)

        if not file_exists:
            writer.writeheader()

        # Fill missing keys with empty values
        for col in keys:
            if col not in data.keys():
                data.update({col: ''})
        
        row_to_add = {i: data[i] for i in keys}
        writer.writerow(row_to_add)


def process_image(image):
    width, height = image.size
    encoding = local_processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    outputs = local_model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    textUnderConsider = []
    true_predictions2 = true_predictions.copy()
    true_boxes2 = true_boxes.copy()
    for k, j in zip(true_predictions2, true_boxes2):
        if k == 'O' or j == [0, 0, 0, 0]:
            true_predictions.remove(k)
            true_boxes.remove(j)
        else:
            textUnderConsider.append([k, j])

    result = extract_text_from_boxes(image, textUnderConsider)

    extractedWords = []
    extractedTags = []
    for i, j in result:
        extractedTags.append(i)
        extractedWords.append(j)

    label_word_pairs = extract_label_word_pairs(extractedWords, extractedTags)

    data = {}
    for i in label_word_pairs:
        data.update(i)

    keys = ['BILLER', 'BILLER_ADDRESS', 'INVOICE_NUMBER', 'BILLER_POST_CODE', 'INVOICE_DATE', 'ABN', 'DUE_DATE', 'SUBTOTAL', 'GST', 'TOTAL']

    write_to_csv(data, keys)

    draw_predictions_on_image(image, true_predictions, true_boxes, label2color)

    return image

def batch_process(input_folder, output_folder,username,identifier):
    if os.path.exists("./data.csv"):
        os.remove("./data.csv")
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(input_path) as img:
                
                processed_img = process_image(img)
                processed_img.save(output_path)
                upload_blob("documentdataextractor.appspot.com", input_path, username+"/"+identifier+"/input/"+filename)
                upload_blob("documentdataextractor.appspot.com", output_path, username+"/"+identifier+"/output/"+filename)

                os.remove(input_path)
                os.remove(output_path)
    
    upload_blob("documentdataextractor.appspot.com", "./data.csv", username+"/"+identifier+"/data.csv")

    print("[Processing complete] Output saved to ", username+"/"+identifier)

    