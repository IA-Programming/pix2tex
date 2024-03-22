import streamlit as st
from PIL import Image
from latex import LatexOCR

def main():
    st.title("Image to LaTeX Formula Parser")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Perform LaTeX OCR on the uploaded image
        latex_formula = process_image(uploaded_image)

        # Display LaTeX formula
        st.subheader("LaTeX Formula:")
        st.text(latex_formula)

        # Display parsed Markdown
        parsed_md = parse_to_md(latex_formula)
        st.subheader("Parsed Markdown:")
        st.latex(f"\n{latex_formula}\n")

def process_image(image):
    # Perform LaTeX OCR on the image
    img = Image.open(image)
    model = LatexOCR()
    latex_formula = model(img)
    return latex_formula

def parse_to_md(latex_formula):
    # You can implement your own logic to parse LaTeX to Markdown
    # Here's a simple example for demonstration purposes
    parsed_md = f"**Parsed Formula:** *{latex_formula}*"
    return parsed_md

if __name__ == "__main__":
    main()
