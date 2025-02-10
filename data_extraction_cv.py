import requests
import json
import pytesseract
from PIL import Image
import re
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Initialize an empty DataFrame to store the extracted data
df = pd.DataFrame(columns=['name', 'company', 'phone number', 'email'])

# Set your OpenRouter API key
OPENROUTER_API_KEY = "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"
YOUR_SITE_URL = 'your_website_url'  # Optional, can be omitted
YOUR_APP_NAME = 'Card Data Extractor App'  # Optional, can be omitted

# Function to call OpenRouter with the Google Gemini model for entity extraction
def extract_entities_with_openrouter(text):
    prompt = f"""
    Extract the following information from the given text of a business card:
    - Name (if any)
    - Company (if any)
    - Phone numbers (if any)
    - Email addresses (if any)
    
    Text:
    {text}

    Provide the output in this format:
    Name: <Name>
    Company: <Company>
    Phone numbers: <Phone Numbers>
    Email addresses: <Email Addresses>
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": YOUR_SITE_URL,  # Optional, if your app has a URL
            "X-Title": YOUR_APP_NAME  # Optional, name of your app for ranking
        },
        data=json.dumps({
            "model": "google/gemini-flash-1.5-8b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error with OpenRouter API: {response.status_code}, {response.text}")
        return None

# Function to extract phone numbers and emails using regular expressions (fallback for GPT results)
def extract_contact_info(text):
    phone_numbers = re.findall(r'\+?\d{1,4}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', text)
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return phone_numbers, emails

# Function to upload an image and process it
def upload_image():
    # Ask the user to select a file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return

    try:
        # Load the image
        image = Image.open(file_path)

        # Perform OCR using Pytesseract
        text = pytesseract.image_to_string(image).strip().replace('\n', ' ')
        print(f"OCR Extracted Text:\n{text}")

        # Extract entities using OpenRouter (Google Gemini model)
        gpt_output = extract_entities_with_openrouter(text)

        if gpt_output:
            print(f"OpenRouter Extracted Entities:\n{gpt_output}")
        else:
            print("Failed to extract entities using OpenRouter. Proceeding with regex fallback.")

        # Use regex extraction as a fallback for phone numbers and emails
        phone_numbers, emails = extract_contact_info(text)

        # Parse OpenRouter (Gemini model) output
        name = re.search(r'Name:\s*(.*)', gpt_output) if gpt_output else None
        company = re.search(r'Company:\s*(.*)', gpt_output) if gpt_output else None
        gpt_phone_numbers = re.search(r'Phone numbers:\s*(.*)', gpt_output) if gpt_output else None
        gpt_emails = re.search(r'Email addresses:\s*(.*)', gpt_output) if gpt_output else None

        # Append the extracted information to the DataFrame
        df.loc[len(df)] = {
            'name': name.group(1).strip() if name else None,
            'company': company.group(1).strip() if company else None,
            'phone number': gpt_phone_numbers.group(1).strip() if gpt_phone_numbers else (phone_numbers[0] if phone_numbers else None),
            'email': gpt_emails.group(1).strip() if gpt_emails else (emails[0] if emails else None)
        }

        # Notify the user of completion
        messagebox.showinfo("Success", "Data extraction complete!")
        print("Data extraction complete.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to save the extracted data to CSV and Excel
def save_data():
    if df.empty:
        messagebox.showwarning("Warning", "No data to save!")
        return

    output_csv = 'output.csv'
    output_excel = 'output.xlsx'

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    # Save the DataFrame to an Excel file
    df.to_excel(output_excel, index=False)

    messagebox.showinfo("Success", f"Files saved as '{output_csv}' and '{output_excel}'.")

# Create the main application window
app = tk.Tk()
app.title("Business Card Data Extractor")
app.geometry("400x200")

# Add buttons to the app
upload_button = tk.Button(app, text="Upload Business Card", command=upload_image)
upload_button.pack(pady=20)

save_button = tk.Button(app, text="Save Extracted Data", command=save_data)
save_button.pack(pady=10)

# Run the application
app.mainloop()
