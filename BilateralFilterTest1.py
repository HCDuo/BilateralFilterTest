import cv2
import numpy as np
from tqdm import tqdm

class ImageProcessor:
    @staticmethod
    def read_image(file_path: str) -> np.ndarray:
        """
        读取图片
        Args:
        file_path (str): 图片文件的路径
        Returns: np.ndarray: 图片数组，像素值在[0, 255]之间
        """
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def save_image(file_path: str, image: np.ndarray) -> None:
        """
        保存图片
        Args:
        file_path (str): 保存图片的路径
        image (np.ndarray): 图片数组，像素值在[0, 255]之间
        Returns: None
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image)

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float, std_dev: float) -> np.ndarray:
        """
        给图片添加高斯噪声
        Args:
        image (np.ndarray): 输入的图片数组，像素值在[0, 255]之间
        mean (float): 高斯分布的均值
        std_dev (float): 高斯分布的标准差
        Returns: np.ndarray: 添加了高斯噪声后的图片数组，像素值在[0, 255]之间
        """
        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def add_salt_and_pepper_noise(image: np.ndarray, noise_density: float) -> np.ndarray:
        """
        给图片添加椒盐噪声

        Args:
        image (np.ndarray): 输入的图片数组，像素值在[0, 255]之间
        noise_density (float): 噪声密度，即噪声点占图片总像素点数的比例

        Returns: np.ndarray: 添加了椒盐噪声后的图片数组，像素值在[0, 255]之间
        """
        noise = np.random.rand(*image.shape)
        noisy_image = np.copy(image)
        noisy_image[noise < noise_density / 2] = 0
        noisy_image[noise > 1 - noise_density / 2] = 255
        return noisy_image


class BilateralFilter:
    def __init__(self, image: np.ndarray, spatial_radius: int, color_sigma: float, space_sigma: float) -> None:
        """
        初始化双边滤波器对象
        Args:
        image (np.ndarray): 输入的图片数组，像素值在[0, 255]之间
        spatial_radius (int): 像素邻域半径
        color_sigma (float): 颜色空间滤波器的sigma值
        space_sigma (float): 空间中滤波器的sigma值
        Returns: None
         """
        self.image = image
        self.spatial_radius = spatial_radius
        self.color_sigma = color_sigma
        self.space_sigma = space_sigma

    def apply(self) -> np.ndarray:
        """
        对图片进行双边滤波
        Returns:
        filtered_image: 双边滤波后的图片数组，像素值在[0, 255]之间
        """
        image = self.image
        filtered_image = np.zeros_like(image)
        pad_size = self.spatial_radius
        image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

        for i in tqdm(range(pad_size, image.shape[0] - pad_size)):
            for j in range(pad_size, image.shape[1] - pad_size):

                filtered_value = np.zeros(image.shape[2])
                weight_sum = 0.0

                for k in range(i - pad_size, i + pad_size + 1):
                    for l in range(j - pad_size, j + pad_size + 1):
                        color_diff = np.linalg.norm(image[k, l] - image[i, j])
                        spatial_diff = np.linalg.norm([k - i, l - j])
                        weight = np.exp(-color_diff ** 2 / (2 * self.color_sigma ** 2) - spatial_diff ** 2 / (
                                    2 * self.space_sigma ** 2))
                        filtered_value += weight * image[k, l]
                        weight_sum += weight

                filtered_image[i - pad_size, j - pad_size] = filtered_value / weight_sum

        return filtered_image


if __name__ == "__main__":
    # 读取图片
    original_image = ImageProcessor.read_image('test.jpg')

    # 添加高斯噪声
    gaussian_noisy_image = ImageProcessor.add_gaussian_noise(original_image, mean=0, std_dev=30)

    # 添加椒盐噪声
    salt_and_pepper_noisy_image = ImageProcessor.add_salt_and_pepper_noise(original_image, noise_density=0.05)

    # 对噪声图片进行双边滤波
    bf = BilateralFilter(gaussian_noisy_image, spatial_radius=5, color_sigma=30, space_sigma=50)
    gaussian_noisy_filtered_image = bf.apply()
    # 对噪声图片进行双边滤波
    bf = BilateralFilter(salt_and_pepper_noisy_image, spatial_radius=5, color_sigma=30, space_sigma=50)
    salt_and_pepper_filtered_image = bf.apply()

    # 保存结果
    ImageProcessor.save_image('D:\\study\\junior\\digital_image_processing\\BilateralFilterTest\\gaussian_noise.jpg', gaussian_noisy_image)
    ImageProcessor.save_image('D:\\study\\junior\\digital_image_processing\\BilateralFilterTest\\gaussian_noise_filtered.jpg', gaussian_noisy_filtered_image)
    ImageProcessor.save_image('D:\\study\\junior\\digital_image_processing\\BilateralFilterTest\\salt_and_pepper_noise.jpg', salt_and_pepper_noisy_image)
    ImageProcessor.save_image('D:\\study\\junior\\digital_image_processing\\BilateralFilterTest\\salt_and_pepper_noise_filtered.jpg', salt_and_pepper_filtered_image)