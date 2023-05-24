# importing required modules
from PyPDF2 import PdfReader
 
# creating a pdf reader object
#reader = PdfReader('insurance_documents/health.pdf')
reader = PdfReader('./mortgage_documents/LD2239.pdf')
 
# printing number of pages in pdf file
print(len(reader.pages))
 
# getting a specific page from the pdf file
page = reader.pages[0]
 
# extracting text from page
text = page.extract_text()
print(text)
