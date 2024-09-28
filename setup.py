from setuptools import setup, find_packages

setup(
    name='cancer_chatbot',
    version='0.1',
    description='This is a chatbot created to answer all your questions related to cancer and cancer research from various journal articles such as PubMed, Biorxiv.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Akshay P R, Kiran K T',
    author_email='akshaypr314159@gmail.com',
    url='https://github.com/Dorcatz123/Biorxiv_chatbot.git',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'langchain',
        'langchain_openai',
        'langchain_community',
        'faiss',
        'langchain_core'

    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  # Specify the Python version your project supports
)
