from setuptools import setup, find_packages

setup(
    name='cancer_researcher_ai',
    version='0.2.1',
    description='This is a chatbot created to answer all your resaerch questions related to cancer from various journal articles such as PubMed, Biorxiv.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Akshay P R, Kiran K T',
    author_email='akshaypr314159@gmail.com',
    url='https://github.com/Dorcatz123/cancer_chatbot.git',
    packages=find_packages(),
    package_data={'cancer_chatbot': ['*.csv']},
    include_package_data = True,
    install_requires=[
        'numpy',
        'pandas',
        'langchain',
        'langchain_openai',
        'langchain_community',
        'faiss-cpu',
        'langchain_core',
        'tqdm'

    ],
    entry_points={
        'console_scripts': [
            'cancer_researcher_ai=cancer_researcher_ai.main:main',  # Define the console command and entry point
        ],
    },


    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify the Python version your project supports
)
