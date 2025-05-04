from google.cloud import documentai, firestore, storage, vision
import fitz  # PyMuPDF
import os
from urllib.parse import quote
from dotenv import load_dotenv
import time
from google.api_core.exceptions import GoogleAPICallError, RetryError

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")

def get_blob(bucket, file_name):
    """Get blob with multiple filename format attempts"""
    attempts = [
        file_name,
        quote(file_name, safe="()"),
        file_name.replace(" ", "_"),
        file_name.replace(" ", "%20")
    ]

    for attempt in attempts:
        blob = bucket.blob(attempt)
        if blob.exists():
            return blob
    raise FileNotFoundError(f"File not found in bucket. Tried: {attempts}")

def process_pdf(bucket_name, file_name):
    """Process PDF with Document AI"""
    print(f" Starting Document AI processing for: {file_name}")

    docai_client = documentai.DocumentProcessorServiceClient()
    storage_client = storage.Client()
    db = firestore.Client()
    bucket = storage_client.bucket(bucket_name)

    if not bucket.exists():
        raise ValueError(f"Bucket {bucket_name} does not exist")

    try:
        blob = get_blob(bucket, file_name)
        processor_name = docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

        gcs_source = documentai.GcsDocument(
            gcs_uri=f"gs://{bucket_name}/{blob.name}",
            mime_type="application/pdf"
        )

        input_config = documentai.BatchDocumentsInputConfig(
            gcs_documents=documentai.GcsDocuments(documents=[gcs_source])
        )

        output_config = documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                gcs_uri=f"gs://{bucket_name}/processed/",
                field_mask="text,pages"
            )
        )

        request = documentai.BatchProcessRequest(
            name=processor_name,
            input_documents=input_config,
            document_output_config=output_config,
        )

        operation = docai_client.batch_process_documents(request=request)
        print(" Waiting for Document AI processing to complete...")
        operation.result(timeout=600)

        metadata = documentai.BatchProcessMetadata(operation.metadata)
        for status in metadata.individual_process_statuses:
            if status.status.code != 0:
                error_msg = f"Document AI Error: {status.status.message}"
                print(f" {error_msg}")
                raise RuntimeError(error_msg)

        output_blobs = list(bucket.list_blobs(prefix="processed/"))
        if not output_blobs:
            raise FileNotFoundError("No output files found in processed/ directory")

        total_pages = 0
        base_name = file_name.replace(".pdf", "")

        for blob in output_blobs:
            if not blob.name.endswith(".json") or base_name not in blob.name:
                continue

            print(f" Processing output shard: {blob.name}")
            doc = documentai.Document.from_json(
                blob.download_as_bytes(),
                ignore_unknown_fields=True
            )

            for page in doc.pages:
                total_pages += 1
                text_content = doc.text[
                    page.layout.text_anchor.text_segments[0].start_index:
                    page.layout.text_anchor.text_segments[0].end_index
                ]

                db.collection("pdf_text").add({
                    "source_file": file_name,
                    "page": total_pages,
                    "content": text_content,
                    "dimensions": {
                        "width": page.dimension.width,
                        "height": page.dimension.height
                    },
                    "processor": PROCESSOR_ID,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })

        print(f" Successfully processed {total_pages} pages")
        return True

    except (GoogleAPICallError, RetryError) as e:
        print(f" Document AI API error: {str(e)}")
        return False
    except Exception as e:
        print(f" Unexpected error in process_pdf: {str(e)}")
        return False
def extract_images_from_pdf(pdf_path, output_folder):
    """Extract images from PDF and save as PNG/JPG"""
    import fitz  # PyMuPDF
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    images = []
    # Iterate through pages
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        
        # Extract each image
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Determine file extension
            ext = base_image["ext"]
            if ext not in ["jpeg", "jpg", "png"]:
                ext = "png"  # Default to PNG for other formats
                
            # Create output path
            output_path = f"{output_folder}/{os.path.basename(pdf_path)}_p{page_num}_i{img_index}.{ext}"
            
            # Save the image
            with open(output_path, "wb") as f:
                f.write(image_bytes)
                
            images.append(output_path)
    
    return images
    
def analyze_image_with_vision(image_content):
    """Analyze image using Vision API"""
    vision_client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)

    features = [
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
        vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION)
    ]

    response = vision_client.annotate_image({
        'image': image,
        'features': features
    })

    description_parts = []

    if response.label_annotations:
        labels = [label.description for label in response.label_annotations[:5]]
        description_parts.append(f"Contains: {', '.join(labels)}")

    if response.localized_object_annotations:
        objects = [obj.name for obj in response.localized_object_annotations[:3]]
        if objects:
            description_parts.append(f"Objects: {', '.join(objects)}")

    if response.text_annotations:
        text = response.text_annotations[0].description.replace('\n', ' ')[:100]
        if text:
            description_parts.append(f"Text: {text}")

    return " | ".join(description_parts) if description_parts else None

def extract_images(bucket_name, file_name):
    """Extract images from PDF with Vision API descriptions"""
    print(f" Starting image extraction for: {file_name}")

    storage_client = storage.Client()
    db = firestore.Client()
    bucket = storage_client.bucket(bucket_name)

    try:
        blob = get_blob(bucket, file_name)
        doc = fitz.open(stream=blob.download_as_bytes())

        image_count = 0
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_content = base_image["image"]

                    vision_description = analyze_image_with_vision(image_content)
                    description = vision_description if vision_description else f"Image from {file_name} on page {page_num + 1}"

                    image_ext = base_image["ext"]
                    image_name = f"{file_name}_p{page_num}_i{img_index}.{image_ext}"
                    image_blob = bucket.blob(f"extracted_images/{image_name}")

                    image_blob.upload_from_string(
                        image_content,
                        content_type=f'image/{image_ext}'
                    )

                    # Generate public URL
                    public_url = f"https://storage.googleapis.com/{bucket_name}/{image_blob.name}"

                    db.collection("pdf_images_new").add({
                        "source_file": file_name,
                        "page": page_num + 1,
                        "image_index": img_index,
                        "image_path": image_blob.name,
                        "public_url": public_url,
                        "description": description,
                        "dimensions": {
                            "width": base_image["width"],
                            "height": base_image["height"]
                        },
                        "format": image_ext,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })

                    image_count += 1
                    print(f" Processed image {img_index + 1} on page {page_num + 1}")

                except Exception as img_error:
                    print(f" Error processing image {img_index} on page {page_num}: {str(img_error)}")

        print(f" Extracted {image_count} images with descriptions from {len(doc)} pages")
        return True

    except Exception as e:
        print(f" Error in extract_images: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF Processor with Document AI and Image Extraction"
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="PDF file name in bucket (can contain spaces)"
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Enable image extraction"
    )

    args = parser.parse_args()

    start_time = time.time()
    process_success = process_pdf(args.bucket, args.file)

    if args.extract_images and process_success:
        extract_success = extract_images(args.bucket, args.file)

    elapsed = time.time() - start_time
    print(f" Total processing time: {elapsed:.2f} seconds")