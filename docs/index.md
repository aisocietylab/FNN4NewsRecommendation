---
layout: project
title: "Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks"
project:
  authors:
    - name: "Kevin Innerebner"
      url: "https://online.tugraz.at/tug_online/visitenkarte.show_vcard?pPersonenId=1E34A1A962FA4625&pPersonenGruppe=3"
    - name: "Stephan Bartl"
    - name: "Markus Reiter-Haas"
      url: "https://iseratho.github.io/"
    - name: "Elisabeth Lex"
      url: "https://elisabethlex.info/"
  institution: "AI for Society Lab"
  conference: "ECIR'26, IR-for-Good-Track"

  links:
    - label: Proceedings
      icon: ai ai-springer
      url: "https://link.springer.com/chapter/10.1007/978-3-032-21324-2_30"
    - label: Paper
      icon: fas fa-file-pdf
      url: "https://arxiv.org/pdf/2601.04019.pdf"
    - label: Code
      icon: fab fa-github
      url: "https://github.com/aisocietylab/FNN4NewsRecommendation"
    - label: arXiv
      icon: ai ai-arxiv
      url: "https://arxiv.org/abs/2601.04019"
    - label: Slides
      icon: fas fa-person-chalkboard
      url: "./assets/pdfs/FNN-ECIR26-presentation.pdf"


  teaser:
    image: "/assets/images/MethodDesign.png"
    alt_text: "Diagram illustrating a fuzzy neural network architecture for a transparent news recommender system. The image shows the preprocessing of user and article features into fuzzy sets, followed by a network of logic nodes (AND/OR layers) that model human-readable rules. The network architecture includes regularization techniques (L1 and orthogonality) and is trained using Binary Cross-Entropy (BCE) loss to predict article clicks. The goal is to provide interpretable rules for editorial decision-making while accurately predicting user click behavior based on behavioral data."
    caption: "The Fuzzy Neural Network architecture for a transparent news recommender system. The network learns human-readable rules from behavioral data to predict article clicks, enabling editorial decision-making based on interpretable patterns in news consumption."
    url: "https://github.com/aisocietylab/FNN4NewsRecommendation"

  abstract: "News recommender systems are increasingly driven by black-box models, offering little transparency for editorial decision-making. In this work, we introduce a transparent recommender system that uses fuzzy neural networks to learn human-readable rules from behavioral data for predicting article clicks. By extracting the rules at configurable thresholds, we can control rule complexity and thus, the level of interpretability. We evaluate our approach on two publicly available news datasets (i.e., MIND and EB-NeRD) and show that we can accurately predict click behavior compared to several established baselines, while learning human-readable rules. Furthermore, we show that the learned rules reveal news consumption patterns, enabling editors to align content curation goals with target audience behavior."
  bibtex: |
    @inproceedings{innerebnerModelingBehavioralPatterns2026,
      address = {Cham},
      title = {Modeling {Behavioral} {Patterns} in {News} {Recommendations} {Using} {Fuzzy} {Neural} {Networks}},
      isbn = {978-3-032-21324-2},
      doi = {10.1007/978-3-032-21324-2_30},
      language = {en},
      booktitle = {Advances in {Information} {Retrieval}},
      publisher = {Springer Nature Switzerland},
      author = {Innerebner, Kevin and Bartl, Stephan and Reiter-Haas, Markus and Lex, Elisabeth},
      editor = {Campos, Ricardo and Jatowt, Adam and Lan, Yanyan and Aliannejadi, Mohammad and Bauer, Christine and MacAvaney, Sean and Anand, Avishek and Ren, Zhaochun and Verberne, Suzan and Bai, Nan and Mansoury, Masoud},
      year = {2026},
      keywords = {Fuzzy Logic, News Consumption, Rule Learning, Transparent Recommender Systems, User Behavior Modeling},
      pages = {384--397},
    }
---

## Acknowledgments

*This research was funded in whole or in part by the Austrian Science Fund (FWF) 10.55776/COE12.*