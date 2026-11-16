# Credits and source provenance

Tell - Tee to Three - AI + Finance, Veritas AI Hackathon 2026.

This credits register records resources used by the project. Main-resource links were checked against primary publishers on November 15, 2026; all 60 selected Commons file-page URLs also returned HTTP 200 in a separate reachability check. It does not grant image-redistribution rights. Receipt images, generated receipt assets and trained weights are not included in the development repository.

## Data and benchmark sources

| Resource | Use in Tell | Source and licence scope |
|---|---|---|
| [GPT4o-Receipt, Scam.AI](https://huggingface.co/datasets/Scam-AI/gpt4o-receipt) | 776 older-generator AI receipt images, train/validation | Dataset card: CC BY-NC-SA 4.0; gated non-commercial research access. The dataset card identifies Zhang et al., [GPT4o-Receipt: A Dataset and Human Study for AI-Generated Document Forensics](https://arxiv.org/abs/2603.11442). The downloaded subset contains AI images, not the full paper's authentic-receipt set. |
| [CORD v2, NAVER CLOVA](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | 1,000 real receipt images, train/validation | Dataset card: CC BY 4.0. Original [CORD repository and citation](https://github.com/clovaai/cord), Park et al., CORD: A Consolidated Receipt Dataset for Post-OCR Parsing. |
| [Wikimedia Commons](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) | 389 retained train/validation images and 60 selected real receipts (50 evaluated, 10 spares) | Individual licences apply; the Commons website's general licence is not a blanket licence for its images. Training-file creator/licence/page records are in [ATTRIBUTION_commons.csv](ATTRIBUTION_commons.csv). Selected receipt source pages are listed below, without receipt pixels or personal information from their contents. |
| [WildReceipt, OpenMMLab](https://github.com/open-mmlab/mmocr/blob/main/dataset_zoo/wildreceipt/metafile.yml) | 1,739 annotated real images added for v2; upstream train/test partitions used as detector train/validation | The primary dataset metafile lists licence type and link as **N/A**. Dataset image rights are therefore **unspecified**, not verified Apache-2.0. The MMOCR code licence does not establish a licence for all receipt photos. Do not redistribute these images or describe their licence as cleared. Original dataset paper: Sun et al., [Spatial Dual-Modality Graph Reasoning for Key Information Extraction](https://arxiv.org/abs/2103.14470). |
| Project-generated receipts | Matched AI twins and unrelated v2 candidates generated for this project | Generated through OpenAI's built-in image-generation tool, orchestrated by coding agents. They are labelled AI and kept private. Generator model identity is not independently recorded consistently enough to claim a precise version. They were not purchased from a stock-image collection. |

## Pretrained model

- [EfficientNet-B0, `timm/efficientnet_b0.ra_in1k`](https://huggingface.co/timm/efficientnet_b0.ra_in1k): the local `timm` default configuration for `efficientnet_b0` resolves to this ImageNet-1k pretrained checkpoint. Model card: Apache-2.0. The project replaces the classifier and fine-tunes it; ImageNet was used through the pretrained weights, not downloaded as a Tell training dataset.
- [timm / PyTorch Image Models](https://github.com/huggingface/pytorch-image-models), Ross Wightman and contributors: architecture implementation and pretrained model loading.
- EfficientNet architecture: Tan and Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), linked from the checkpoint's model card.

## Software and infrastructure

| Resource | Project role |
|---|---|
| [Python](https://docs.python.org/3/) | Application, data scripts and standard library |
| [PyTorch](https://pytorch.org/) and [torchvision](https://docs.pytorch.org/vision/stable/index.html) | Model training/inference and image transforms |
| [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [pandas](https://pandas.pydata.org/) | Arrays, statistics/calibration and split/result tables |
| [scikit-learn](https://scikit-learn.org/stable/) | Logistic-regression baselines, preprocessing and evaluation metrics |
| [Pillow](https://pillow.readthedocs.io/en/stable/) | Image decoding, metadata stripping, normalization and contact sheets |
| [Tesseract OCR](https://tesseract-ocr.github.io/) and [pytesseract](https://github.com/madmaze/pytesseract) | OCR bookkeeping-checklist baseline; English OCR support used |
| [ImageHash](https://github.com/JohannesBuchner/imagehash) | Perceptual-hash duplicate rule in the checklist baseline |
| [Hugging Face Hub client](https://huggingface.co/docs/huggingface_hub/index) and [Apache Arrow / PyArrow](https://arrow.apache.org/docs/python/) | Dataset/model retrieval and CORD Parquet extraction |
| [Gradio](https://gradio.app/) | Local receipt-screening interface |
| [Matplotlib](https://matplotlib.org/) | Notebook/Grad-CAM visualizations; scientific drafting where used |
| [Jupyter nbformat](https://nbformat.readthedocs.io/en/latest/) and [nbclient](https://nbclient.readthedocs.io/en/latest/) | Notebook creation and local execution checks |
| [Google Colab](https://colab.google/) | Target environment for the included training notebook; reported final training ran locally on Apple MPS |
| [ReportLab](https://docs.reportlab.com/), [pypdf](https://pypdf.readthedocs.io/en/stable/) and [Poppler](https://poppler.freedesktop.org/) | Writeup PDF generation, structural checks and rendered visual inspection |

Software credit does not imply endorsement by any maintainer. Dependencies retain their own licences; no single dataset or model licence covers the entire project.

## AI assistance disclosure

We used [OpenAI Codex](https://openai.com/codex/), [OpenAI image generation](https://developers.openai.com/api/docs/guides/image-generation), [Anthropic Claude](https://claude.com/product/overview) and [Google Gemini](https://deepmind.google/models/gemini/).

## Selected Commons receipt source credits

The 389-row training attribution CSV does **not** include the selected heldout/spare IDs. The table below supplies their separate source records. Creator names are drawn from cached Commons metadata where available, otherwise the selection record; the linked original file page remains authoritative. No receipt text, customer/card/staff details, local raw-image paths or generation prompts are copied here. Before displaying any image, check its source page's current attribution/licence requirements and redact the eight privacy-marked originals.

| ID | Source file | Creator (metadata) | Licence (metadata) | Role |
|---|---|---|---|---|
| r0041 | [Polish supermarket receipt.jpg](https://commons.wikimedia.org/wiki/File:Polish_supermarket_receipt.jpg) | Hippietrail | CC BY-SA 4.0 | test |
| r0102 | [Żabka in Wrocław 2019 05 чек.jpg](https://commons.wikimedia.org/wiki/File:%C5%BBabka_in_Wroc%C5%82aw_2019_05_%D1%87%D0%B5%D0%BA.jpg) | Nickispeaki | CC BY-SA 4.0 | test |
| r0562 | [Paragon fiskalny- 14.05.2025.jpg](https://commons.wikimedia.org/wiki/File:Paragon_fiskalny-_14.05.2025.jpg) | Aw58 | CC BY 4.0 | test |
| r0592 | [Paragon fiskalny - 26.08.2024.jpg](https://commons.wikimedia.org/wiki/File:Paragon_fiskalny_-_26.08.2024.jpg) | Aw58 | CC BY-SA 4.0 | test |
| r0283 | [AVIA Veendam receipt, Winschoten (2019) 01.jpg](https://commons.wikimedia.org/wiki/File:AVIA_Veendam_receipt%2C_Winschoten_%282019%29_01.jpg) | Donald Trung Quoc Don (Chữ Hán: 徵國單) - Wikimedia Commons - © CC BY-SA 4.0 International. (Want to use this image?) Original publication 📤: --Donald Trung 『徵國單』 (No Fake News 💬) (WikiProject Numismatics 💴) (Articles 📚) 19:36, 1 May 2019 (UTC) | CC BY-SA 4.0 | test |
| r0303 | [De Heerlijckheid van Winschoten receipt, Winschoten (2019) 01.jpg](https://commons.wikimedia.org/wiki/File:De_Heerlijckheid_van_Winschoten_receipt%2C_Winschoten_%282019%29_01.jpg) | Donald Trung Quoc Don (Chữ Hán: 徵國單) - Wikimedia Commons - © CC BY-SA 4.0 International. (Want to use this image?) Original publication 📤: --Donald Trung 『徵國單』 (No Fake News 💬) (WikiProject Numismatics 💴) (Articles 📚) 16:11, 1 May 2019 (UTC) | CC BY-SA 4.0 | test |
| r0388 | [Medikamente Die Grenze receipt, Winschoten (2019) 01.jpg](https://commons.wikimedia.org/wiki/File:Medikamente_Die_Grenze_receipt%2C_Winschoten_%282019%29_01.jpg) | Donald Trung Quoc Don (Chữ Hán: 徵國單) - Wikimedia Commons - © CC BY-SA 4.0 International. (Want to use this image?) Original publication 📤: --Donald Trung 『徵國單』 (No Fake News 💬) (WikiProject Numismatics 💴) (Articles 📚) 16:00, 1 May 2019 (UTC) | CC BY-SA 4.0 | test |
| r0123 | [Alves receipt (2019) 01.jpg](https://commons.wikimedia.org/wiki/File:Alves_receipt_%282019%29_01.jpg) | Donald Trung Quoc Don (Chữ Hán: 徵國單) - Wikimedia Commons. (Want to use this image?) | CC BY-SA 4.0 | test |
| r0209 | [A-Receipt-Cascade-bakery.jpg](https://commons.wikimedia.org/wiki/File:A-Receipt-Cascade-bakery.jpg) | HQA02330 | CC BY-SA 4.0 | test |
| r0240 | [Receipt of CamelMart 220917.jpg](https://commons.wikimedia.org/wiki/File:Receipt_of_CamelMart_220917.jpg) | Yauchi | CC BY-SA 4.0 | test |
| r0261 | [ガストの伝票20201024.jpg](https://commons.wikimedia.org/wiki/File:%E3%82%AC%E3%82%B9%E3%83%88%E3%81%AE%E4%BC%9D%E7%A5%A820201024.jpg) | 経済特区 | CC BY-SA 4.0 | test |
| r0324 | [領収書 レギュラーガソリン 10.92L 2011-03-20.jpg](https://commons.wikimedia.org/wiki/File:%E9%A0%98%E5%8F%8E%E6%9B%B8_%E3%83%AC%E3%82%AE%E3%83%A5%E3%83%A9%E3%83%BC%E3%82%AC%E3%82%BD%E3%83%AA%E3%83%B3_10.92L_2011-03-20.jpg) | contri | CC BY-SA 4.0 | test |
| r0326 | [黒三角 (35025724572).jpg](https://commons.wikimedia.org/wiki/File:%E9%BB%92%E4%B8%89%E8%A7%92_%2835025724572%29.jpg) | 芳蘭 徐芳蘭 | CC BY 2.0 | test |
| r0119 | [Watsons Dawulun Store e-invoice AF41230438.jpg](https://commons.wikimedia.org/wiki/File:Watsons_Dawulun_Store_e-invoice_AF41230438.jpg) | Solomon203 | CC BY-SA 4.0 | test |
| r0750 | [9x9 Stationery Proficient e-invoice 20170415 face.jpg](https://commons.wikimedia.org/wiki/File:9x9_Stationery_Proficient_e-invoice_20170415_face.jpg) | Solomon203 | CC BY-SA 4.0 | test |
| r0752 | [TW IKEA uniform invoice 20130216.jpg](https://commons.wikimedia.org/wiki/File:TW_IKEA_uniform_invoice_20130216.jpg) | Solomon203 | CC BY-SA 3.0 | test |
| r0015 | [China IMG 4103 (29451995060).jpg](https://commons.wikimedia.org/wiki/File:China_IMG_4103_%2829451995060%29.jpg) | Kuruman from Tokyo, Japan | CC BY 2.0 | test |
| r0127 | [American Coffee tickets (April 8, 2025).jpg](https://commons.wikimedia.org/wiki/File:American_Coffee_tickets_%28April_8%2C_2025%29.jpg) | 茅野ふたば | CC BY-SA 4.0 | test |
| r0151 | [SZ 深圳 Shenzhen 寶安中心 BaoAn 創業一路 Chuangye 1st Road mall AEON Supermarket bill December 2024 R12S 01.jpg](https://commons.wikimedia.org/wiki/File:SZ_%E6%B7%B1%E5%9C%B3_Shenzhen_%E5%AF%B6%E5%AE%89%E4%B8%AD%E5%BF%83_BaoAn_%E5%89%B5%E6%A5%AD%E4%B8%80%E8%B7%AF_Chuangye_1st_Road_mall_AEON_Supermarket_bill_December_2024_R12S_01.jpg) | TaiZiThee 832 | CC0 | test |
| r0024 | [HK SYP 西環 Sai Ying Pun 德輔道西 308 Des Voeux Road West 食神麗宮酒家 Chinese Banquet Seafood Restaurant bill receipt January 2026 N13P 02.jpg](https://commons.wikimedia.org/wiki/File:HK_SYP_%E8%A5%BF%E7%92%B0_Sai_Ying_Pun_%E5%BE%B7%E8%BC%94%E9%81%93%E8%A5%BF_308_Des_Voeux_Road_West_%E9%A3%9F%E7%A5%9E%E9%BA%97%E5%AE%AE%E9%85%92%E5%AE%B6_Chinese_Banquet_Seafood_Restaurant_bill_receipt_January_2026_N13P_02.jpg) | EVOW Sahgyune Paiem KAASEI 02238 | CC0 | test |
| r0071 | [Zeppelinwurst Einkaufsbeleg Metzgerei Ebert.jpg](https://commons.wikimedia.org/wiki/File:Zeppelinwurst_Einkaufsbeleg_Metzgerei_Ebert.jpg) | scanned by Eva K. | Public domain | test |
| r0198 | [Real receipt, Oude Pekela (2019) 01.jpg](https://commons.wikimedia.org/wiki/File:Real_receipt%2C_Oude_Pekela_%282019%29_01.jpg) | Donald Trung Quoc Don (Chữ Hán: 徵國單) - Wikimedia Commons - © CC BY-SA 4.0 International. (Want to use this image?) Original publication 📤: --Donald Trung 『徵國單』 (No Fake News 💬) (WikiProject Numismatics 💴) (Articles 📚) 22:29, 30 December 2019 (UTC) | CC BY-SA 4.0 | test |
| r0202 | [Schlecker-Quittung-2009.jpg](https://commons.wikimedia.org/wiki/File:Schlecker-Quittung-2009.jpg) | Riquix | CC BY-SA 4.0 | test |
| r0108 | [Grocery Store Receipt in Vienna.jpg](https://commons.wikimedia.org/wiki/File:Grocery_Store_Receipt_in_Vienna.jpg) | Grandmaster Huon | Public domain | test |
| r0194 | [Italian supermarket receipt showing value-added tax (IVA) categories.jpg](https://commons.wikimedia.org/wiki/File:Italian_supermarket_receipt_showing_value-added_tax_%28IVA%29_categories.jpg) | Sgroey | CC BY-SA 4.0 | test |
| r0053 | [Save Mart recipt 2010-10-23.jpg](https://commons.wikimedia.org/wiki/File:Save_Mart_recipt_2010-10-23.jpg) | Amin Eshaiker | Public domain | test |
| r0691 | [L'Oro di Napoli - January 2025 - Sarah Stierch 22.jpg](https://commons.wikimedia.org/wiki/File:L%27Oro_di_Napoli_-_January_2025_-_Sarah_Stierch_22.jpg) | Missvain | CC0 | test |
| r0695 | [Target Receipt 5-12-2018.jpg](https://commons.wikimedia.org/wiki/File:Target_Receipt_5-12-2018.jpg) | 123TheBusHonolulu696969 | CC BY-SA 4.0 | test |
| r0693 | [Smith's Food and Drug 190 recipt (2012-01-02).jpg](https://commons.wikimedia.org/wiki/File:Smith%27s_Food_and_Drug_190_recipt_%282012-01-02%29.jpg) | Amin Eshaiker | Public domain | test |
| r0697 | [Walgreens Receipt with RX number 2.jpg](https://commons.wikimedia.org/wiki/File:Walgreens_Receipt_with_RX_number_2.jpg) | MCruz (WMF) | CC BY-SA 4.0 | test |
| r0674 | [TK Maxx, Trafford Retail Park receipt (2013-05-12).jpg](https://commons.wikimedia.org/wiki/File:TK_Maxx%2C_Trafford_Retail_Park_receipt_%282013-05-12%29.jpg) | TK Maxx | Public domain | test |
| r0753 | [We spent what?? (9532631438).jpg](https://commons.wikimedia.org/wiki/File:We_spent_what%3F%3F_%289532631438%29.jpg) | William J Sisti from Morri | CC BY-SA 2.0 | test |
| r0065 | [Tower records Dublin receipt 2016.jpg](https://commons.wikimedia.org/wiki/File:Tower_records_Dublin_receipt_2016.jpg) | Joehawkins | CC BY-SA 4.0 | test |
| r0048 | [Reciept.jpg](https://commons.wikimedia.org/wiki/File:Reciept.jpg) | Doug OLES | CC BY-SA 4.0 | test |
| r0097 | [Tickets de caisse de la crêperie Tante Lucie.jpg](https://commons.wikimedia.org/wiki/File:Tickets_de_caisse_de_la_cr%C3%AAperie_Tante_Lucie.jpg) | Crêperie Tante Lucie | Public domain | test |
| r0604 | [-sablazosmadrid 'Café & Te'-- "en fin de semana cobramos el desayuno por separado" -haztefan pero no vayas (6225790880).jpg](https://commons.wikimedia.org/wiki/File:-sablazosmadrid_%27Caf%C3%A9_%26_Te%27--_%22en_fin_de_semana_cobramos_el_desayuno_por_separado%22_-haztefan_pero_no_vayas_%286225790880%29.jpg) | Stéphane M. Grueso from Ma | CC BY-SA 2.0 | test |
| r0027 | [Kvīts.jpg](https://commons.wikimedia.org/wiki/File:Kv%C4%ABts.jpg) | Hippietrail | CC0 | test |
| r0680 | [Чек за купівлю масок медичних для учасників Вікімарафону 2022 в бібліотеці.jpg](https://commons.wikimedia.org/wiki/File:%D0%A7%D0%B5%D0%BA_%D0%B7%D0%B0_%D0%BA%D1%83%D0%BF%D1%96%D0%B2%D0%BB%D1%8E_%D0%BC%D0%B0%D1%81%D0%BE%D0%BA_%D0%BC%D0%B5%D0%B4%D0%B8%D1%87%D0%BD%D0%B8%D1%85_%D0%B4%D0%BB%D1%8F_%D1%83%D1%87%D0%B0%D1%81%D0%BD%D0%B8%D0%BA%D1%96%D0%B2_%D0%92%D1%96%D0%BA%D1%96%D0%BC%D0%B0%D1%80%D0%B0%D1%84%D0%BE%D0%BD%D1%83_2022_%D0%B2_%D0%B1%D1%96%D0%B1%D0%BB%D1%96%D0%BE%D1%82%D0%B5%D1%86%D1%96.jpg) | Ninohka-17 | CC BY-SA 4.0 | test |
| r0607 | [20190221 124822 Teremok Receipt Квитанция.jpg](https://commons.wikimedia.org/wiki/File:20190221_124822_Teremok_Receipt_%D0%9A%D0%B2%D0%B8%D1%82%D0%B0%D0%BD%D1%86%D0%B8%D1%8F.jpg) | Rakoon | CC0 | test |
| r0746 | [Účtenka Sklářský sklep.jpg](https://commons.wikimedia.org/wiki/File:%C3%9A%C4%8Dtenka_Skl%C3%A1%C5%99sk%C3%BD_sklep.jpg) | Emited by Becex Trade s.r.o. Scanned by ŠJů (cs:ŠJů) | Public domain | test |
| r0666 | [A receipt in Thai from a restaurant in Chiang Mai (2019).jpg](https://commons.wikimedia.org/wiki/File:A_receipt_in_Thai_from_a_restaurant_in_Chiang_Mai_%282019%29.jpg) | Chainwit. | CC BY 4.0 | test |
| r0692 | [Một tờ hóa đơn ăn uống ở Việt Nam.jpg](https://commons.wikimedia.org/wiki/File:M%E1%BB%99t_t%E1%BB%9D_h%C3%B3a_%C4%91%C6%A1n_%C4%83n_u%E1%BB%91ng_%E1%BB%9F_Vi%E1%BB%87t_Nam.jpg) | Martin Lewison from Forest  | CC BY 2.0 | test |
| r0093 | [Receipt from Lawson in South Tangerang, Indonesia with QRIS CPM (2025-09-02).jpg](https://commons.wikimedia.org/wiki/File:Receipt_from_Lawson_in_South_Tangerang%2C_Indonesia_with_QRIS_CPM_%282025-09-02%29.jpg) | VulcanSphere | CC BY 4.0 | test |
| r0081 | [Bill Vishwanand 2026-01-28.jpg](https://commons.wikimedia.org/wiki/File:Bill_Vishwanand_2026-01-28.jpg) | Sakurakat | CC0 | test |
| r0078 | [20200803 190954 things places in south korea IMG 8318.jpg](https://commons.wikimedia.org/wiki/File:20200803_190954_things_places_in_south_korea_IMG_8318.jpg) | Choi Kwang-mo | CC0 | test |
| r0074 | [0B83181C-3DCA-4008-837D-5ED65hdekkektguufnvnnf.jpg](https://commons.wikimedia.org/wiki/File:0B83181C-3DCA-4008-837D-5ED65hdekkektguufnvnnf.jpg) | Helleiken | CC BY-SA 4.0 | test |
| r0011 | [BRA — Rio Grande do Norte — Natal (recibo Salosino).jpg](https://commons.wikimedia.org/wiki/File:BRA_%E2%80%94_Rio_Grande_do_Norte_%E2%80%94_Natal_%28recibo_Salosino%29.jpg) | see image | Public domain | test |
| r0196 | [Minimarket-avi-30-september-invoice.jpg](https://commons.wikimedia.org/wiki/File:Minimarket-avi-30-september-invoice.jpg) | Rakoon | CC BY-SA 3.0 | test |
| r0476 | [KSU WUGN HANGOUT RECEIPT1.jpg](https://commons.wikimedia.org/wiki/File:KSU_WUGN_HANGOUT_RECEIPT1.jpg) | Obapeace | CC BY-SA 4.0 | test |
| r0472 | [Caloocan Malabon Areas 20.jpg](https://commons.wikimedia.org/wiki/File:Caloocan_Malabon_Areas_20.jpg) | Zarate123 | CC BY-SA 4.0 | test |
| r0128 | [Bọn Bini Supermarket receipt (2019) 01.jpg](https://commons.wikimedia.org/wiki/File:B%E1%BB%8Dn_Bini_Supermarket_receipt_%282019%29_01.jpg) | Donald Trung Quoc Don (Chữ Hán: 徵國單) - Wikimedia Commons - © CC BY-SA 4.0 International. (Want to use this image?) Original publication 📤: --Donald Trung 『徵國單』 (No Fake News 💬) (WikiProject Numismatics 💴) (Articles 📚) 14:38, 10 April 2019 (UTC) | CC BY-SA 4.0 | spare |
| r0321 | [美秀美術館 (23898009427).jpg](https://commons.wikimedia.org/wiki/File:%E7%BE%8E%E7%A7%80%E7%BE%8E%E8%A1%93%E9%A4%A8_%2823898009427%29.jpg) | 芳蘭 徐芳蘭 | CC BY 2.0 | spare |
| r0567 | [Paragon fiskalny - 2024.11.25.jpg](https://commons.wikimedia.org/wiki/File:Paragon_fiskalny_-_2024.11.25.jpg) | Aw58 | CC BY 4.0 | spare |
| r0446 | [Tamoil klantenbon, Oude Pekela (2019) 01.jpg](https://commons.wikimedia.org/wiki/File:Tamoil_klantenbon%2C_Oude_Pekela_%282019%29_01.jpg) | Donald Trung Quoc Don (Chữ Hán: 徵國單) - Wikimedia Commons - © CC BY-SA 4.0 International. (Want to use this image?) Original publication 📤: --Donald Trung 『徵國單』 (No Fake News 💬) (WikiProject Numismatics 💴) (Articles 📚) 10:14, 13 April 2019 (UTC) | CC BY-SA 4.0 | spare |
| r0157 | [SZ 深圳 Shenzhen 龍崗 Longgang 坂田 Bantian 城中村 Village shop 開心購生活超市 Happy Go Life Supermarket bill May 2024 R12S 30.jpg](https://commons.wikimedia.org/wiki/File:SZ_%E6%B7%B1%E5%9C%B3_Shenzhen_%E9%BE%8D%E5%B4%97_Longgang_%E5%9D%82%E7%94%B0_Bantian_%E5%9F%8E%E4%B8%AD%E6%9D%91_Village_shop_%E9%96%8B%E5%BF%83%E8%B3%BC%E7%94%9F%E6%B4%BB%E8%B6%85%E5%B8%82_Happy_Go_Life_Supermarket_bill_May_2024_R12S_30.jpg) | Guw Foiursanm Longum | CC0 | spare |
| r0185 | [MediaMarkt-Quittung-2011.jpg](https://commons.wikimedia.org/wiki/File:MediaMarkt-Quittung-2011.jpg) | Riquix | CC BY-SA 4.0 | spare |
| r0665 | [99 cents only store receipt 2021.jpg](https://commons.wikimedia.org/wiki/File:99_cents_only_store_receipt_2021.jpg) | Downtowngal | CC BY-SA 4.0 | spare |
| r0113 | [Receipt-woolworth.jpg](https://commons.wikimedia.org/wiki/File:Receipt-woolworth.jpg) | Daryldaryl | CC BY-SA 4.0 | spare |
| r0677 | [Магазин Сільпо 10 березня 2022 року в Чернігові 10 Чек.jpg](https://commons.wikimedia.org/wiki/File:%D0%9C%D0%B0%D0%B3%D0%B0%D0%B7%D0%B8%D0%BD_%D0%A1%D1%96%D0%BB%D1%8C%D0%BF%D0%BE_10_%D0%B1%D0%B5%D1%80%D0%B5%D0%B7%D0%BD%D1%8F_2022_%D1%80%D0%BE%D0%BA%D1%83_%D0%B2_%D0%A7%D0%B5%D1%80%D0%BD%D1%96%D0%B3%D0%BE%D0%B2%D1%96_10_%D0%A7%D0%B5%D0%BA.jpg) | Nickispeaki | CC BY-SA 4.0 | spare |
| r0052 | [SK Korea tour 首爾 supermarket official receipt computer print out in Korean language only July-2013.JPG](https://commons.wikimedia.org/wiki/File:SK_Korea_tour_%E9%A6%96%E7%88%BE_supermarket_official_receipt_computer_print_out_in_Korean_language_only_July-2013.JPG) | Woodarelaisku | CC BY-SA 3.0 | spare |
