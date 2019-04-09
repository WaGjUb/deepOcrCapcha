## Ref: https://code-maven.com/create-images-with-python-pil-pillow
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os

# img size
im_w = 200
im_h = 100 

## character quantity
qnt = 5 

# fonts path
fontpath = './fonts'

# letters domain config 
lower = (97, 122+1)
upper = (65,90+1)
nums = (48,57+1)
iterate = [lower, upper, nums]
domain = []

# domain letters for captcha
for i in iterate:
    for j in range(i[0],i[1]):
        domain.append(chr(j))

def gen_background():
    arr = (np.random.rand(2,2,3) * 200) + 55
    image = Image.fromarray(arr.astype('uint8')).convert('RGB')
    image = image.resize((im_w,im_h))
    image = image.filter(ImageFilter.GaussianBlur(1))
    return image

def random_text(forced=None):
    # randomize indexes of domain
    ret = []
    for i in range(0,qnt):
        ret.append(domain[int(np.random.rand(1)*(len(domain)))])
    # force to have an character (it's good to make sure that the our model will see all characters at least n times)
    if forced:
        ret[int(np.random.rand(1)*(len(ret)))] = domain[forced]
    return ''.join(ret)

def drawText(draw, text, fnt_size=(35,50)):
    size = (im_w/(qnt))-5
    fontSize = int(np.random.rand(1)*(fnt_size[1]-fnt_size[0])) + fnt_size[0]

    for idx, l in enumerate(text):
        allfonts = os.listdir(fontpath)
        fnt_name = fontpath + '/' + allfonts[int(np.random.rand(1)*len(allfonts))]
        fnt = ImageFont.truetype(fnt_name,fontSize)
        text_w, text_h = draw.textsize(l, font=fnt)

        y = (np.random.rand(1)*(im_h-(fontSize)))
        # check char height is in bound
        if y - (text_h/2)  < 0:
            y = y + abs(y-(text_h))
        if y + (text_h/2) > im_h:
            y = y - ((y+text_h) - im_h)

        x = (np.random.rand(1)*size) + (size*idx)
        # check char width is in bound 
        if x - (text_w/2) < size*idx:
            x = (size*idx) + (text_w/2)
        if x + (text_w/2) > size*(idx+1):
            x = (size*(idx+1)) - (text_w)
        draw.text((x,y), l, font=fnt, fill=tuple((np.random.rand(3)*150).astype('uint8')))

def gen_img():
    back = gen_background()
    draw = ImageDraw.Draw(back)
    label = random_text()
    drawText(draw, label)
    draw.line((np.random.rand(1)*im_w,0) + (np.random.rand(1)*im_w, im_h), fill=0, width=1)
    #drawText(draw, '12345')
    #draw.text((50,50),random_text(), fill=(255,255,0))
    return (back, label)

def save_new(path='./generated/'):
    img, label = gen_img()
    img.save(path+label+'.png')
