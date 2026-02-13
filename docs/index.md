---
layout: project
project:
  authors:
    - name: "Kevin Innerebner"
    - name: "Stephan Bartl"
    - name: "Markus Reiter-Haas"
      url: "https://iseratho.github.io/"
    - name: "Elisabeth Lex"
      url: "https://elisabethlex.info/"
  institution: "AI for Society Lab"
  conference: "ECIR'26, IR-for-Good-Track"

  links:
    - label: Paper
      icon: fas fa-file-pdf
      url: "https://arxiv.org/pdf/2601.04019.pdf"
    - label: Code
      icon: fab fa-github
      url: "https://github.com/aisocietylab/FNN4NewsRecommendation"
    - label: arXiv
      icon: ai ai-arxiv
      url: "https://arxiv.org/abs/2601.04019"


  teaser:
    image: "/assets/images/MethodDesign.png"
    alt_text: "Diagram illustrating a fuzzy neural network architecture for a transparent news recommender system. The image shows the preprocessing of user and article features into fuzzy sets, followed by a network of logic nodes (AND/OR layers) that model human-readable rules. The network architecture includes regularization techniques (L1 and orthogonality) and is trained using Binary Cross-Entropy (BCE) loss to predict article clicks. The goal is to provide interpretable rules for editorial decision-making while accurately predicting user click behavior based on behavioral data."
    caption: "The Fuzzy Neural Network architecture for a transparent news recommender system. The network learns human-readable rules from behavioral data to predict article clicks, enabling editorial decision-making based on interpretable patterns in news consumption."
    url: "https://github.com/aisocietylab/FNN4NewsRecommendation"

  abstract: "News recommender systems are increasingly driven by black-box models, offering little transparency for editorial decision-making. In this work, we introduce a transparent recommender system that uses fuzzy neural networks to learn human-readable rules from behavioral data for predicting article clicks. By extracting the rules at configurable thresholds, we can control rule complexity and thus, the level of interpretability. We evaluate our approach on two publicly available news datasets (i.e., MIND and EB-NeRD) and show that we can accurately predict click behavior compared to several established baselines, while learning human-readable rules. Furthermore, we show that the learned rules reveal news consumption patterns, enabling editors to align content curation goals with target audience behavior."
  bibtex: |
    @misc{innerebner2026modelingbehavioralpatternsnews,
          title={Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks}, 
          author={Kevin Innerebner and Stephan Bartl and Markus Reiter-Haas and Elisabeth Lex},
          year={2026},
          eprint={2601.04019},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2601.04019}, 
    }
---