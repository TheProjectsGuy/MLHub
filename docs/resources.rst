External Resources for ML
=========================

Links, blogs, courses, people, etc. in the fields of AI, ML, CV, NLP, etc. that I want to keep at one place (for my reference).

.. contents:: Table of contents
    :depth: 4

Websites
--------

Daily papers
^^^^^^^^^^^^

#. `HuggingFace Papers <https://huggingface.co/papers>`_
#. `PapersWithCode <https://paperswithcode.com/>`_ (also see `other portals <https://portal.paperswithcode.com/>`_)
#. `arxiv-sanity <https://arxiv-sanity-lite.com/>`_
#. `AIModels.fyi <https://notes.aimodels.fyi/>`_ blog notes (paper summaries)

Paper search

#. `Litmaps <https://www.litmaps.com/>`_
#. `Semantic Scholar <https://www.semanticscholar.org/>`_
#. `Google Scholar <https://scholar.google.com/>`_

Conferences

#. `AI Deadlines <https://aideadlin.es/>`_: A `PapersWithCode Repository <https://github.com/paperswithcode/ai-deadlines>`_ that keeps track of upcoming conferences in AI related subject areas

AI Summaries

.. warning:: 
    AI generated summaries might not always be factual (or have high utility).

#. `Bing Chat on Microsoft Edge <https://www.reddit.com/r/bing/s/SOvYIzjMwd>`_: Open PDF in Edge, click Bing Chat, ask it questions about the PDF
#. `arxiv-summary <https://www.arxiv-summary.com/>`_
#. `scisummary <https://scisummary.com/>`_

Blogs
^^^^^

#. `The AISummer <https://theaisummer.com/>`_: Getting started with AI and many concepts pages as `articles <https://theaisummer.com/learn-ai/>`_

    * See page on `attention <https://theaisummer.com/attention/>`_, `NLP transformers <https://theaisummer.com/transformer/>`_, and `ViT transformer <https://theaisummer.com/transformer/>`_
    * See page on `NeRF and InstantNGP <https://theaisummer.com/nerf/>`_
    * See page on `Diffusion models <https://theaisummer.com/diffusion-models/>`_

#. `Lilian Weng's Blog <https://lilianweng.github.io/>`_

    * See `Transformer version family <https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/>`_
    * See `Prompt engineering <https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/>`_
    * See overview of `LLM powered autonomous agents <https://lilianweng.github.io/posts/2023-06-23-agent/>`_

Podcasts
^^^^^^^^

#. `Lex Fridman's Podcast <https://lexfridman.com/podcast/>`_

Projects
--------

Interesting projects in the wild

#. `HuggingFace Projects Documentation <https://huggingface.co/docs>`_: A collection of awesome community projects

    * `Transformers <https://huggingface.co/docs/transformers/index>`_: Different transformer implementations
    * `timm <https://huggingface.co/docs/timm/index>`_: SOTA computer vision implementations
    * `Hub <https://huggingface.co/docs/hub/index>`_: Models, datasets, etc. at one place
    * `Tokenizers <https://huggingface.co/docs/tokenizers/index>`_: Tokenizers for production and research
    * `Diffusers <https://huggingface.co/docs/diffusers/index>`_: Diffusion algorithms

#. `Kornia <https://kornia.readthedocs.io/en/latest/>`_: Computer vision algorithms (AI centric)
#. `PyG - PyTorch Geometric <https://pyg.org/>`_: Geometric deep learning on PyTorch
#. `Numba <https://numba.pydata.org/>`_: JIT compilation for making python code faster (even has CUDA acceleration and parallel for loops)
#. `CuPy <https://cupy.dev/>`_ and `PyCUDA <https://documen.tician.de/pycuda/>`_: NVIDIA CUDA in Python
#. `HMMLearn <https://hmmlearn.readthedocs.io/en/latest/index.html>`_: Unsupervised learning and inference of Hidden Markov Models
#. `XGBoost <https://xgboost.readthedocs.io/en/stable/>`_: Optimized gradient boosting library that is efficient, flexible, and portable
#. `LightGBM <https://lightgbm.readthedocs.io/en/latest/index.html>`_: Gradient boosting framework that uses tree-based learning algorithms
#. `Ray <https://www.ray.io/>`_: Scaling AI workloads (data, training, hyperparameter tuning, RL, serving, etc.). From `AnyScale Platform <https://www.anyscale.com/platform>`_ team.
#. Projects on large language models

    * `Lightning-AI/lit-gpt <https://github.com/Lightning-AI/lit-gpt>`_: Implementation of SOTA open-source LLMs with quantization and LoRA like enhancements
    * `LLaMA Index <https://www.llamaindex.ai/>`_: LLMs on your own data
    * `LangChain <https://python.langchain.com/>`_: Build applications powered by language models
    * `tiktoken <https://github.com/openai/tiktoken>`_: OpenAI's BPE tokenizer

#. `spaCy <https://spacy.io/>`_: NLP tool
#. `Radiant Earth <https://radiant.earth/>`_: Earth observation data (geo-spatial informatics)
#. `Acme <https://dm-acme.readthedocs.io/en/latest/>`_: RL components and agents by Google DeepMind
#. `ICESat-2 <https://icesat-2.gsfc.nasa.gov/>`_: Ice, Cloud and land Elevation Satellite-2 (geo-spatial informatics)
#. `MLHub CLI <https://mlhub.readthedocs.io/en/latest/>`_: Command line framework for various ML models (not related to this project)
#. `AutoML <https://www.automl.org/>`_: Neural architecture search (NAS) and hyperparameter selection/optimization

Startups
--------

#. `ArtPark Ignite <https://www.artpark.in/startup/ignite/>`_: Venture-building program for AI and Robotics from ARTPARK@IISc

Books
-----

#. `Ian Goodfellow - Deep Learning book <https://www.deeplearningbook.org/>`_

Courses
-------

AI and Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^

#. `Stanford CS229 - Machine Learning - Prof. Anand Avati <http://cs229.stanford.edu/>`_

    * Stanford's Machine Learning course. There are five modules; supervised learning: linear and logistic regression, classification, linear models, generative learning, kernel methods, and support vector machines (SVMs); deep learning: neural networks and back propagation; generalisation and regularisation: complexity bounds and model selection; unsupervised learning: clustering, expectation maximisation (EM) algorithms (ELBO), VAEs, PCA, Independent Component Analysis, self-supervised learning (SSL) and foundation models; reinforcement learning: decision processes, policies, linear quadratic regulation (LQR), differential dynamic programming (DDP), linear quadratic gaussians (LQG), policy gradients. Main course design by Andrew Ng.
    * Related: 

        * `Stanford CS230 - Deep Learning - Andrew Ng <https://cs230.stanford.edu/>`_: `YouTube playlist - Autumn 2018 <https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb>`_

    * Links: `Website <http://cs229.stanford.edu/>`_ (`SEE Page <https://see.stanford.edu/Course/CS229>`_, `Stanford page <https://online.stanford.edu/courses/cs229-machine-learning>`_), `CS229 Fall 2023-24 Syllabus <https://docs.google.com/spreadsheets/d/1sEu4ygD5HWxaqjvbR2nsjvG6NBoW5tRW/edit>`_, `Course Notes by Andrew Ng <https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf>`_, `YouTube Playlist - Spring 2023 <https://youtube.com/playlist?list=PLoROMvodv4rNyWOpJg_Yh4NSqI4Z4vOYy>`_, `YouTube Playlist - Autumn 2018 <https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&si=abStj_Mu__Xu_vIb>`_

#. `NYU - Deep Learning - SP21 <https://cds.nyu.edu/deep-learning/>`_

    * Deep learning course at NYU from Yann LeCun and Alfredo Canziani
    * Links: `Course Docs - Spring 2020 <https://atcold.github.io/NYU-DLSP20/>`_ (major release, other `didactics <https://atcold.github.io/didactics>`_), `YouTube Playlist - Spring 2020 <https://www.youtube.com/playlist?list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq>`_, `GitHub - Spring 2021 <https://github.com/Atcold/NYU-DLSP21>`_

#. `Stanford CS231n - Deep Learning for Computer VIsion - Fei Fei Li <http://cs231n.stanford.edu/>`_

    * Links: `YouTube Playlist <https://youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv>`_, `Course website <https://cs231n.github.io/>`_

#. `CMU - 11-785 Introduction to Deep Learning <https://deeplearning.cs.cmu.edu/F22/index.html>`_
#. `CMU - 16-825 - Learning for 3D Vision - Spring 2023 <https://learning3d.github.io/>`_

    * `Course GitHub (Assignments) <https://github.com/learning3d/>`_, `GitHub (Submissions) <https://github.com/Zoe0123/16-825-Learning-for-3D-Vision/tree/main>`_

#. `Cornell Tech CS 5785 - Applied Machine Learning <https://classes.cornell.edu/browse/roster/FA23/class/CS/5785>`_

    * Links: `YouTube playlist - Fall 2020 <https://www.youtube.com/playlist?list=PL2UML_KCiC0UlY7iCQDSiGDMovaupqc83>`_, `GitHub - Notebooks and Slides <https://github.com/kuleshov/cornell-cs5785-2020-applied-ml>`_

#. `MIT 6.S192: Deep Learning for Art, Aesthetics, and Creativity <https://ali-design.github.io/deepcreativity/>`_

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PLCpMvp7ftsnIbNwRnQJbDNRqO6qiN3EyH>`_

#. `MIT - Introduction to Deep Learning <http://introtodeeplearning.com/>`_

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI>`_

#. `Stanford CS25 - Transformers United <https://web.stanford.edu/class/cs25/>`_

    * Links: `YouTube Playlist - Cases <https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM>`_

#. `UC Berkeley - Full Stack Deep Learning <https://fullstackdeeplearning.com/course/>`_

    * Links: `YouTube Playlist - Spring 2021 <https://www.youtube.com/playlist?list=PL1T8fO7ArWlcWg04OgNiJy91PywMKT2lv>`_

#. `University of Tubingen - Statistical Machine Learning - Summer 2020 <https://www.tml.cs.uni-tuebingen.de/teaching/2020_statistical_learning/>`_

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC>`_

#. `University of Tubingen - Introduction to Machine Learning - Winter 2020/21 <https://www.youtube.com/playlist?list=PL05umP7R6ij35ShKLDqccJSDntugY4FQT>`_ 

    * Links: `Dmitry Kobak's Blog - Slides <https://dkobak.github.io/>`_

#. `UC Berkeley - CS294-158-SP20 - Deep Unsupervised Learning Spring 2020 <https://sites.google.com/view/berkeley-cs294-158-sp20/home>`_

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PLwRJQ4m4UJjPiJP3691u-qWwPGVKzSlNP>`_

#. `Michigan - EECS 498.008 / 598.008 - Deep Learning for Computer Vision - Winter 2022 <https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/>`_ 

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r>`_

Geometric Deep Learning
"""""""""""""""""""""""

#. `UvA - An Introduction to Group Equivariant Deep Learning <https://uvagedl.github.io/>`_

    * Part of `Geometric Deep Learning <https://geometricdeeplearning.com/>`_ series from University of Amsterdam. Contains lecture videos on group theory, steerable group convolutions, and equivariant graph neural networks. Also has Colab assignments.

#. `UPenn - Graph Neural Networks - ESE 5140 <https://gnn.seas.upenn.edu/>`_ 

    * GNNs (lectures and labs/assignments). Overview of GNNs from `NVIDIA <https://blogs.nvidia.com/blog/2022/10/24/what-are-graph-neural-networks/>`_, `distill <https://distill.pub/2021/gnn-intro/>`_


Reinforcement Learning
""""""""""""""""""""""

#. `Stanford CS234 - Reinforcement Learning - Emma Brunskill <https://web.stanford.edu/class/cs234/>`_

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u>`_

#. `UC Berkeley CS 285 - Deep Reinforcement Learning <https://rail.eecs.berkeley.edu/deeprlcourse/>`_ 

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc>`_

#. `UC Berkeley CS 294 - Deep Reinforcement Learning (Fall 2015) <https://rll.berkeley.edu/deeprlcourse-fa15/>`_

    * Links: `YouTube Playlist - Foundations of Deep RL - Pieter Abbeel <https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0>`_


Computer Vision
^^^^^^^^^^^^^^^

#. `University of Tubingen - Computer Vision - Prof. Dr. Andreas Geiger <https://uni-tuebingen.de/en/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/>`_

    * Introduction and history of computer vision. Photogrammetry, image sensing pipeline, structure-from-motion, bundle adjustment, stereo reconstruction, probabilistic graphical models, optical flow, shape from shading, stereo, coordinate based networks, image recognition, semantic segmentation, object detection, self-supervised learning, and other advanced topics (compositional models, human body models, deepfakes, etc.). University of Tubingen Computer Vision course by Prof. Dr. Andreas Geiger.
    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_>`_, `Public Material: Slides and exercises <https://drive.google.com/drive/folders/17YkOlItn9PycNb5bT_O4nVlavlX0_VKQ>`_

Natural Language Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. `CMU - CS 11-737 Multilingual NLP - Spring 2022 <https://www.phontron.com/class/multiling2022/index.html>`_

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PL8PYTP1V4I8BhCpzfdKKdd1OnTfLcyZr7>`_

#. `CMU - CS 11-711 - Advanced NLP - Fall 2022 <https://www.phontron.com/class/anlp2022/>`_

    * Links: `YouTube Playlist <https://www.youtube.com/playlist?list=PL8PYTP1V4I8D0UkqW2fEhgLrnlDW9QK7z>`_

#. `Stanford CS224U: Natural Language Understanding <https://web.stanford.edu/class/cs224u/>`_

    * Links: `GitHub <https://github.com/cgpotts/cs224u>`_, `YouTube Playlist <https://www.youtube.com/playlist?list=PLoROMvodv4rPt5D0zs3YhbWSZA8Q_DyiJ>`_

#. `UMass - CS685 - Advanced Natural Language Processing - Spring 2023 <https://people.cs.umass.edu/~miyyer/cs685/>`_

    * Links: `YouTube Playlist - Fall 2020 <https://www.youtube.com/playlist?list=PLWnsVgP6CzadmQX6qevbar3_vDBioWHJL>`_

YouTube Playlists
-----------------

#. `Andrej Karpathy - Neural Networks: Zero to Hero <https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ>`_
#. `Samuel Albanie - Foundation Models <https://www.youtube.com/playlist?list=PL9t0xVFP90GD8hox0KipBkJcLX_C3ja67>`_
#. `GCP -  Making Friends with Machine Learning <https://www.youtube.com/playlist?list=PLRKtJ4IpxJpDxl0NTvNYQWKCYzHNuy2xG>`_
#. `HuggingFace Course YouTube Playlist <https://www.youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o>`_

    * Links: `All HF Courses <https://huggingface.co/learn>`_, `HF NLP Course <https://huggingface.co/learn/nlp-course>`_, `HF Audio Course <https://huggingface.co/learn/audio-course>`_, `HF Deep RL Course <https://huggingface.co/learn/deep-rl-course>`_

#. `Jeremy Howard - Practical Deep Learning for Coders 2022 <https://www.youtube.com/playlist?list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU>`_
#. `MLOps - Machine Learning Engineering for Production <https://www.youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK>`_

Communities
-----------

Some communities you can follow

#. `ML Collective <https://mlcollective.org/>`_: ML research opportunities, collaboration, and mentorship

People
------

#. `Geoffrey E. Hinton <https://www.cs.toronto.edu/~hinton/>`_, `Yann LeCun <http://yann.lecun.org/ex/>`_, and `Yoshua Bengio <https://yoshuabengio.org/>`_: Founders of modern deep learning (received the turing award for it in 2018)
#. `Jurgen Schmidhuber <https://people.idsia.ch/~juergen/>`_ (IDSAI, Swiss): LSTM
#. `Jitendra Malik <https://people.eecs.berkeley.edu/~malik/>`_ (UC Berkeley, Meta): Computer vision and AI
#. `Leonidas J Guibas <https://profiles.stanford.edu/leonidas-guibas>`_ (Stanford): 3D computer vision backbones (PointNet).
#. `Abhinav Gupta <https://www.cs.cmu.edu/~abhinavg/>`_ (CMU RI): Computer Vision and AI
#. `Sergey Levine <https://people.eecs.berkeley.edu/~svlevine/>`_ (UC Berkeley): Reinforcement Learning for Robotics
#. `Dhruv Batra <https://faculty.cc.gatech.edu/~dbatra/>`_ (Georgia Tech, Meta): Embodied AI Agents, Robotics
#. `Michael Bronstein <https://www.cs.ox.ac.uk/people/michael.bronstein/>`_ (CS Univ. of Oxford): Geometric deep learning and graph neural networks.
#. `Max Welling <https://staff.fnwi.uva.nl/m.welling/>`_ (Qualcomm UvA): VAEs, graph CNNs
#. `Luca Carlone <https://lucacarlone.mit.edu/>`_ (MIT): SPARK Lab; SLAM and robust perception.
#. `Saurabh Gupta <https://saurabhg.web.illinois.edu/>`_ (UIUC, Meta): Computer vision, robotics, and AI

Follow these folks on social media (for new research)

#. `Dmytro Mishkin <https://dmytro.ai/>`_: Kornia (CV+AI framework), tweets papers
#. `Phil Wang a.k.a. Lucidrains <https://lucidrains.github.io/>`_: Open source contributions on `GitHub <https://github.com/lucidrains>`_
#. `Ahsen Khaliq a.k.a. AK a.k.a. akhaliq <https://twitter.com/_akhaliq>`_: Tweets and HuggingFace papers, Gradio
#. `Aran Komatsuzaki <https://arankomatsuzaki.wordpress.com/about-me/>`_: Tweets papers, LAION and EleutherAI
#. `Mike Young <https://twitter.com/mikeyoung44>`_: Paper summaries
#. `Ryohei Sasaki <https://github.com/rsasaki0109>`_: Research on autonomous driving (LiDAR)
#. `Dr Ronald Clark <https://www.ron-clark.com/>`_ (CS, Oxford): Real time SLAM, bundle adjustment, scene understanding, and motion tracking
#. `Devendra Singh Chaplot <https://devendrachaplot.github.io/>`_ (CMU, FAIR): Visual navigation, object goal navigation, exploration, embodied AI
#. `Dhruv Shah <https://twitter.com/shahdhruv_>`_ (UC Berkeley): Robotics & AI
