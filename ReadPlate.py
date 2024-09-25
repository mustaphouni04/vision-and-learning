import easyocr

def ReadPlate(plate, show=True):
    reader = easyocr.Reader(['es']) # Create an easyocr reader
    result = reader.readtext(plate) # Read text from the cropped image
    text = [i[-2] for i in result] # Extract the text from the result    
    
    if show:
        print('LICENSE PLATE:',text)
    return text


plates = ['path1', 'path2', 'pathn']
output = []

for plate in plates:
    text = ReadPlate(plate)
    output.append(text)
