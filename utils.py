import pytesseract
from PIL import  ImageDraw, ImageFont

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def iob_to_label(label):
    return label

def extract_text_from_boxes(image, boxes):
    extracted_data = []
    # i = 0
    for label, coordinates in boxes:
        roi = image.crop((coordinates[0] - 4, coordinates[1] - 4, coordinates[2] + 4, coordinates[3] + 4))
        # os.makedirs('roi', exist_ok=True)
        # roi.save(f'roi/roi{i}.png')
        # i += 1
        text = pytesseract.image_to_string(roi, config='--psm 6')
        extracted_data.append([label, text.strip()])
    return extracted_data


def extract_label_word_pairs(extractedWords, extractedTags):
    label_word_pairs = []
    current_label = None
    current_word = None

    for word, tag in zip(extractedWords, extractedTags):
        if tag.startswith('B-'):
            if current_label is not None and current_word is not None:
                label_word_pairs.append({current_label: current_word})
            current_label = tag.split('-')[1]
            current_word = word
        elif tag.startswith('I-'):
            if current_label is not None and current_word is not None:
                current_word += " " + word
        else:
            if current_label is not None and current_word is not None:
                label_word_pairs.append({current_label: current_word})
                current_label = None
                current_word = None

    # Handling the last pair
    if current_label is not None and current_word is not None:
        label_word_pairs.append({current_label: current_word})

    return label_word_pairs
def draw_predictions_on_image(image, true_predictions, true_boxes, label2color):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction)
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
