#Output_path should contain an image name with extension like out.bmp

from processing import convert_frame

img_path = input('Img_path: ')
out_path = input('Output_path: ')

frame = img_path

out_img = convert_frame(frame,image=True)

out_img.save(out_path)
print('output image saved to ', out_path)