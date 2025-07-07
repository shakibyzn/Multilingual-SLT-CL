# Continual Learning in Multilingual Sign Language Translation
Official implementation for the NAACL 2025 [paper](https://aclanthology.org/2025.naacl-long.546.pdf): Continual Learning in Multilingual Sign Language Translation

## Data Preparation
We use two feature extraction methods (I3D and S3D) on the datasets below
- [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)
- [How2Sign](https://how2sign.github.io/)
- [SP-10](https://github.com/MLSLT/SP-10)

If you need the extracted features, please send an email to [shakib yazdani](shakibyzn@gmail.com).

#### I3D Feature extraction
Go to [WLASL](https://github.com/dxli94/WLASL/tree/master?tab=readme-ov-file) GitHub and download the I3D pre-trained [weights](https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view). For each dataset, run its associated Python file under `feat-ext/I3D`.


## Citation

If you find our model, data or the overview of data useful for your research, please cite:

```
@inproceedings{yazdani-etal-2025-continual,
    title = "Continual Learning in Multilingual Sign Language Translation",
    author = "Yazdani, Shakib  and
      Genabith, Josef Van  and
      Espa{\~n}a-Bonet, Cristina",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.546/",
    pages = "10923--10938",
    ISBN = "979-8-89176-189-6",
}
```

## Acknowledgements