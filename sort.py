import glob
from os.path 					import splitext, basename
image_paths = glob.glob("test/*.jpg")
xml_paths = glob.glob("output/*.jpg")
img_paths = []
xml = []
print(len(image_paths))
print(len(xml_paths))
for img in image_paths:
    # img = img[0:len(img[0])-5]
    img = splitext(basename(img))[0]
    img_paths.append(img)
for img in xml_paths:
    # img = img[0:len(img[0])-5]
    img = splitext(basename(img))[0]
    xml.append(img)
print(len(img_paths))
print(len(xml))
def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
print(Diff(img_paths,xml))

