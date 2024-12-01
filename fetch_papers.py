import os
import re
import requests
import codecs

# 创建存放论文的目录
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 处理README.md文件中的内容，获取论文信息
def process_readme(file_path):
    with codecs.open(file_path, encoding='utf-8', mode='r') as f:
        lines = f.read().split('\n')

    sections = []  # 存储章节信息
    current_section_path = ''
    section_title_regex = re.compile(r'###(.*)')

    # 处理每一行内容
    for line in lines:
        # 如果是章节标题
        if '###' in line:
            section_title = section_title_regex.search(line).group(1).strip()
            # 清除Windows系统不支持的字符
            section_title = clean_title(section_title)
            # 创建章节目录
            current_section_path = os.path.join('papers', section_title)
            create_directory(current_section_path)
            sections.append((section_title, current_section_path))

        # 如果是包含论文PDF链接的行
        if '[pdf]' in line:
            paper_info = extract_paper_info(line)
            if paper_info:
                paper_title, paper_url = paper_info
                # 下载论文
                download_paper(current_section_path, paper_title, paper_url)

    return sections

# 清理标题中的非法字符
def clean_title(title):
    win_restricted_chars = re.compile(r'[\\/*?:"<>|]')
    return win_restricted_chars.sub("", title)

# 提取论文标题和PDF链接
def extract_paper_info(line):
    match = re.search(r'\*\*(.*?)\*\*.*?\[\[pdf\]\]\((.*?)\)', line)
    if match:
        return match.groups()
    return None

# 下载论文
def download_paper(section_path, paper_title, paper_url):
    paper_path = os.path.join(section_path, paper_title + '.pdf')
    # 如果文件已存在，则跳过下载
    if not os.path.exists(paper_path):
        print(f'正在下载 {paper_title}...')
        try:
            response = requests.get(paper_url)
            response.raise_for_status()  # 检查请求是否成功
            with open(paper_path, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {e}")

# 主函数，组织整个流程
def main():
    # 1. 创建存放论文的根目录
    create_directory('papers')

    # 2. 解析README.md文件，获取论文信息
    process_readme('README.md')

# 执行主函数
if __name__ == "__main__":
    main()
