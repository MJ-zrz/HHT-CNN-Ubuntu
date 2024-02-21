from PIL import Image
 
def image_resize(input_path, output_path, resize_size_list):
    # 打开原始图像文件
    image = Image.open(input_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 调整图像大小为指定的宽度和高度
    width, height = resize_size_list
    cropped_image = image.resize((width, height))
    
    # 保存修改后的图像到输出路径
    cropped_image.save(output_path)


def image_crop(input_path, output_path, crop_size_list):
    # 打开原始图像
    image = Image.open(input_path)
    
    # 需要裁剪的区域
    left, top, right, bottom = crop_size_list
    
    # 根据指定的区域进行裁剪
    cropped_image = image.crop((left, top, right, bottom))
    
    # 保存裁剪后的图像到输出路径
    cropped_image.save(output_path)


