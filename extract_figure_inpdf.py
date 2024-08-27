import fitz  # PyMuPDF
import cv2
import numpy as np
import os


def is_diagram(image):
    """Determine if the image likely contains a diagram using contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours  # Return contours instead of just the count


def extract_diagram_region(image, contours, width_padding=180, height_padding=140):
    """Extract the bounding box for the largest contour with added padding for width and height."""
    if contours:
        # Find the largest contour (assuming it's the diagram)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extend the width of the bounding box
        x = max(0, x - width_padding)  # Prevent going out of bounds on the left
        w = min(
            image.shape[1] - x, w + 2 * width_padding
        )  # Prevent going out of bounds on the right

        # Extend the height of the bounding box
        y = max(0, y - height_padding)  # Prevent going out of bounds at the top
        h = min(
            image.shape[0] - y, h + 2 * height_padding
        )  # Prevent going out of bounds at the bottom

        # Extract the region of interest (ROI)
        diagram_region = image[y : y + h, x : x + w]
        return diagram_region
    return None


def extract_diagrams_from_pdf(pdf_path, output_folder):
    """Extract diagrams from a PDF file."""
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each page
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]

        # Convert page to an image
        pix = page.get_pixmap()

        # Convert pixmap to a NumPy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # If the image has an alpha channel, convert it to BGR
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Check if the image contains a diagram
        contours = is_diagram(img)

        # Extract diagram region if available
        diagram = extract_diagram_region(
            img,
            contours,
            width_padding=180,
            height_padding=140,  # Updated height padding to 140
        )

        # Check if diagram height is at least 300 pixels before saving
        if diagram is not None and diagram.shape[0] >= 300:
            image_filename = os.path.join(
                output_folder, f"diagram_page_{page_number + 1}.jpg"
            )

            # Save the extracted diagram
            cv2.imwrite(image_filename, diagram)
            print(f"Saved diagram: {image_filename}")
        else:
            if diagram is not None:
                print(
                    f"Skipped saving diagram on page {page_number + 1} due to insufficient height."
                )

    print("Diagram extraction completed.")
    pdf_document.close()


# Example usage
pdf_file_path = "example.pdf"  # Replace with your PDF path
output_directory = "output_diagrams"  # Replace with your output directory
extract_diagrams_from_pdf(pdf_file_path, output_directory)
