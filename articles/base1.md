www.nature.com/scientificreports
 OPEN
 An effective detection approach
for phishing websites using URL
and HTML features
 Ali Aljofey1,2, Qingshan Jiang1*, Abdur Rasool1,2, Hui Chen1,2, Wenyin Liu3, Qiang Qu1 &
Yang Wang4
 Today’s growing phishing websites pose significant threats due to their extremely undetectable risk.
They anticipate internet users to mistake them as genuine ones in order to reveal user information
and privacy, such as login ids, pass-words, credit card numbers, etc. without notice. This paper
proposes a new approach to solve the anti-phishing problem. The new features of this approach can
be represented by URL character sequence without phishing prior knowledge, various hyperlink
information, and textual content of the webpage, which are combined and fed to train the XGBoost
classifier. One of the major contributions of this paper is the selection of different new features,
which are capable enough to detect 0-h attacks, and these features do not depend on any third
party services. In particular, we extract character level Term Frequency-Inverse Document Frequency
(TF-IDF) features from noisy parts of HTML and plaintext of the given webpage. Moreover, our
proposed hyperlink features determine the relationship between the content and the URL of a
webpage. Due to the absence of publicly available large phishing data sets, we needed to create
our own data set with 60,252 webpages to validate the proposed solution. This data contains 32,972
benign webpages and 27,280 phishing webpages. For evaluations, the performance of each category
of the proposed feature set is evaluated, and various classification algorithms are employed. From
the empirical results, it was observed that the proposed individual features are valuable for phishing
detection. However, the integration of all the features improves the detection of phishing sites with
significant accuracy. The proposed approach achieved an accuracy of 96.76% with only 1.39% false
positive rate on our dataset, and an accuracy of 98.48% with 2.09% false-positive rate on benchmark
dataset, which outperforms the existing baseline approaches.
 Phishing offenses are increasing, resulting in billions of dollars in  loss1. In these attacks, users enter their critical
(i.e., credit card details, passwords, etc.) to the forged website which appears to be legitimate. The Software-as
a-Service (SaaS) and webmail sites are the most common targets of  phishing2. The phisher makes websites that
look very similar to the benign websites. The phishing website link is then sent to millions of internet users via
emails and other communication media. These types of cyber-attacks are usually activated by emails, instant mes
sages, or phone  calls3. The aim of the phishing attack is not only to steal the victims’ personality, but it can also be
performed to spread other types of malware such as ransomware, to exploit approach weaknesses, or to receive
monetary profits4. According to the Anti-Phishing Working Group (APWG) report in the 3rd Quarter of 2020,
the number of phishing attacks has grown since March, and 28,093 unique phishing sites have been detected
between July to  September2. The average amount demanded during wire transfer Business E-mail Compromise
(BEC) attacks was $48,000 in the third quarter, down from $80,000 in the second quarter and $54,000 in the first.
 Detecting and preventing phishing offenses is a significant challenge for researchers due to the way phish
ers carry out the attack to bypass the existing anti-phishing techniques. Moreover, the phisher can even target
some educated and experienced users by using new phishing scams. Thus, software-based phishing detection
techniques are preferred for fighting against the phishing attack. Mostly available methods for detecting phish
ing attacks are blacklists/whitelists5, natural language  processing6, visual  similarity7, rules8, machine learning
techniques 9,10, etc. Techniques based on blacklists/whitelists fail to detect unlisted phishing sites (i.e. 0-h attacks)
1Shenzhen Key Laboratory for High Performance Data Mining, Shenzhen Institute of Advanced Technology,
Chinese Academy of Sciences, Shenzhen 518055, China. 2Shenzhen College of Advanced Technology, University
of Chinese Academy of Sciences, Beijing 100049, China. 3Department of Computer Science, Guangdong University
of Technology, Guangzhou, China. 4Cloud Computing Center, Shenzhen Institute of Advanced Technology, Chinese
Academy of Sciences, Shenzhen 518055, China. *email: qs.jiang@siat.ac.cn
 Scientific Reports |         (2022) 12:8842
| https://doi.org/10.1038/s41598-022-10841-5
 1
 Vol.:(0123456789)
www.nature.com/scientificreports/
 as well as these methods fail when blacklisted URL is encountered with minor changes. In the machine learning
based techniques, a classification model is trained using various heuristic features (i.e., URL, webpage content,
website traffic, search engine, WHOIS record, and Page Rank) in order to improve detection efficiency. However,
these heuristic features are not warranted to present in all phishing websites and might also present in the benign
websites, which may cause a classification error. Moreover, some of the heuristic features are hard to access and
third-party dependent. Some third-party services (i.e., page rank, search engine indexing, WHOIS etc.) may not
be sufficient to identify phishing websites that are hosted on hacked servers and these websites are inaccurately
identified as benign websites because they are contained in search results. Websites hosted on compromised
servers are usually more than a day old unlike other phishing websites which only take a few hours. Also, these
services inaccurately identify the new benign website as a phishing site due to the lack of domain age. The visual
similarity-based heuristic techniques compare the new website with the pre-stored signature of the website. The
website’s visual signature includes screenshots, font styles, images, page layouts, logos, etc. Thus, these techniques
cannot identify the fresh phishing websites and generate a high false-negative rate (phishing to benign). The URL
based technique does not consider the HTML of the webpage and may misjudge some of the malicious websites
hosted on free or compromised servers. Many existing  approaches11–13 extract hand-crafted URL based features,
e.g., number of dots, presence of special “@”, “#”, “–” symbol, URL length, brand names in URL, position of Top
Level domain, check hostname for IP address, presence of multiple TLDs, etc. However, there are still hurdles
to extracting manual URL features due to the fact that human effort requires time and extra maintenance labor
costs. Detecting and preventing phishing offense is a major defiance for researchers because the scammer car
ries out these offenses in a way that can avoid current anti-phishing methods. Hence, the use of hybrid methods
rather than a single approach is highly recommended by the networks security manager.
 T
 his paper provides an efficient solution for phishing detection that extracts the features from website’s URL
and HTML source code. Specifically, we proposed a hybrid feature set including URL character sequence features
without expert’s knowledge, various hyperlink information, plaintext and noisy HTML data-based features within
the HTML source code. These features are then used to create feature vector required for training the proposed
approach by XGBoost classifier. Extensive experiments show that the proposed anti-phishing approach has
attained competitive performance on real dataset in terms of different evaluation statistics.
 Our anti-phishing approach has been designed to meet the following requirements.
 • High detection efficiency: To provide high detection efficiency, incorrect classification of benign sites as
phishing (false-positive) should be minimal and correct classification of phishing sites (true-positive) should
be high.
 • Real-time detection: The prediction of the phishing detection approach must be provided before exposing
the user’s personal information on the phishing website.
 • Target independent: Due to the features extracted from both URL and HTML the proposed approach can
detect new phishing websites targeting any benign website (zero-day attack).
 • Third-party independent: The feature set defined in our work are lightweight and client-side adaptable, which
do not rely on third-party services such as blacklist/whitelist, Domain Name System (DNS) records, WHOIS
record (domain age), search engine indexing, network traffic measures, etc. Though third-party services may
raise the effectiveness of the detection approach, they might misclassify benign websites if a benign website
is newly registered. Furthermore, the DNS database and domain age record may be poisoned and lead to
false negative results (phishing to benign).
 Hence, a light-weight technique is needed for phishing websites detection adaptable at client side. The
major contributions in this paper are itemized as follows.
 • We propose a phishing detection approach, which extracts efficient features from the URL and HTML of the
given webpage without relying on third-party services. Thus, it can be adaptable at the client side and specify
better privacy.
 • We proposed eight novel features including URL character sequence features (F1), textual content character
level (F2), various hyperlink features (F3, F4, F5, F6, F7, and F14) along with seven existing features adopted
from the literature.
 • We conducted extensive experiments using various machine learning algorithms to measure the efficiency
of the proposed features. Evaluation results manifest that the proposed approach precisely identifies the
legitimate websites as it has a high true negative rate and very less false positive rate.
 • We release a real phishing webpage detection dataset to be used by other researchers on this topic.
 T
 he rest of this paper is structured as follows: The "Related work" section first reviews the related works
about phishing detection. Then the "Proposed approach" section presents an overview of our proposed solution
and describes the proposed features set to train the machine learning algorithms. The "Experiments and result
analysis” section introduces extensive experiments including the experimental dataset and results evaluations.
Furthermore, the "Discussion and limitation" section contains a discussion and limitations of the proposed
approach. Finally, the "Conclusion" section concludes the paper and discusses future work.
 Related work
 T
 his section provides an overview of the proposed phishing detection techniques in the literature. Phishing meth
ods are divided into two categories; expanding the user awareness to distinguish the characteristics of phishing
and benign  webpages14, and using some extra software. Software-based techniques are further categorized into
list-based detection, and machine learning-based detection. However, the problem of phishing is so sophisticated
Scientific Reports |         (2022) 12:8842  |
Vol:.(1234567890)
 https://doi.org/10.1038/s41598-022-10841-5
 2
www.nature.com/scientificreports/
 that there is no definitive solution to efficiently bypass all threats; thus, multiple techniques are often dedicated
to restrain particular phishing offenses.
 List-based detection. List-based phishing detection methods use either whitelist or blacklist-based tech
nique. A blacklist contains a list of suspicious domains, URLs, and IP addresses, which are used to validate if
a URL is fraudulent. Simultaneously, the whitelist is a list of legitimate domains, URLs, and IP addresses used
to validate a suspected URL. Wang et al.15, Jain and  Gupta5 and Han et al.16 use white list-based method for the
detection of suspected URL. Blacklist-based methods are widely used in openly available anti-phishing toolbars,
such as Google safe browsing, which maintains a blacklist of URLs and provides warnings to users once a URL is
considered as phishing. Prakash et al.17 proposed a technique to predict phishing URLs called Phishnet. In this
technique, phishing URLs are identified from the existing blacklisted URLs using the directory structure, equiv
alent IP address, and brand name. Felegyhazi et al.18 developed a method that compares the domain name and
name server information of new suspicious URLs to the information of blacklisted URLs for the classification
process. Sheng et al.19 demonstrated that a forged domain was added to the blacklist after a considerable amount
of time, and approximately 50–80% of the forged domains were appended after the attack was carried out. Since
thousands of deceptive websites are launched every day, the blacklist requires to be updated periodically from its
source. Thus, machine learning-based detection techniques are more efficient in dealing with phishing offenses.
 Machine learning-based detection. Data mining techniques have provided outstanding performance in
many applications, e.g., data security and  privacy20, game  theory21, blockchain systems22, healthcare23, etc. Due
to the recent development of phishing detection methods, various machine learning-based techniques have also
been employed6,9,10,13 to investigate the legality of websites. The effectiveness of these methods relies on feature
collection, training data, and classification algorithm. The feature collection is extracted from different sources,
e.g., URL, webpage content, third party services, etc. However, some of the heuristic features are hard to access
and time-consuming, which makes some machine learning approaches demand high computations to extract
these features.
 Jain and  Gupta24 proposed an anti-phishing approach that extracts the features from the URL and source code
of the webpage and does not rely on any third-party services. Although the proposed approach attained high
accuracy in detecting  phishing webpages, it used a limited dataset (2141 phishing and 1918 legitimate webpages).
T
 he same  authors9 present a phishing detection method that can identify phishing attacks by analyzing the hyper
links extracted from the HTML of the webpage. The proposed method is a client-side and language-independent
solution. However, it entirely depends on the HTML of the webpage and may incorrectly classify the phishing
webpages if the attacker changes all webpage resource references (i.e., Javascript, CSS, images, etc.). Rao and  Pais25
proposed a two-level anti-phishing technique called BlackPhish. At first level, a blacklist of signatures is created
using visual similarity based features (i.e., file names, paths, and screenshots) rather than using blacklist of URLs.
At second level, heuristic features are extracted from URL and HTML to identify the phishing websites which
override the first level filter. In spite of that, the legitimate websites always undergo two-level filtering. In some
researches26 authors used search engine-based mechanism to authenticate the webpage as first-level authentica
tion. In the second level authentication, various hyperlinks within the HTML of the website are processed for
the phishing websites detection. Although the use of search engine-based techniques increases the number of
legitimate websites correctly identified as legitimate, it also increases the number of legitimate websites incorrectly
identified as phishing when newly created authentic websites are not found in the top results of search engine.
Search based approaches assume that genuine website appears in the top search results.
 In a recent study, Rao et al.27 proposed a new phishing websites detection method with word embedding
extracted from plain text and domain specific text of the html source code. They implemented different word
embedding to evaluate their model using ensemble and multimodal techniques. However, the proposed method
is entirely dependent on plain text and domain specific text, and may fail when the text is replaced with images.
Some researchers have tried to identify phishing attacks by extracting different hyperlink relationships from web
pages. Guo et al.28 proposed a phishing webpages detection approach which they called HinPhish. The approach
establishes a heterogeneous information network (HIN) based on domain nodes and loading resources nodes
and establishes three relationships between the four hyperlinks: external link, empty link, internal link and rela
tive link. Then, they applied an authority ranking algorithm to calculate the effect of different relationships and
obtain a quantitative score for each node.
 In Sahingoz et al.6 work, the distributed representation of words is adopted within a specific URL, and then
seven various machine learning classifiers are employed to identify whether a suspicious URL is a phishing
website. Rao et al.13 proposed an anti-phishing technique called CatchPhish. They extracted hand-crafted and
Term Frequency-Inverse Document Frequency (TF-IDF) features from URLs, then trained a classifier on the
features using random forest algorithm. Although the above methods have shown satisfactory performance,
they suffer from the following restrictions: (1) inability to handle unobserved characters because the URLs
usually contain meaningless and unknown words that are not in the training set; (2) they do not consider the
content of the website. Accordingly, some URLs, which are distinctive to others but imitate the legitimate sites,
may not be identified based on URL string. As their work is only based on URL features, which is not enough
to detect the phishing websites. However, we have provided an effective solution by proposing our approach to
this domain by utilizing three different types of features to detect the phishing website more efficiently. Specifi
cally, we proposed a hybrid feature set consisting of URL character sequence, various hyperlinks information,
and textual content-based features.
 Deep learning methods have been used for phishing detection e.g., Convolutional Neural Network (CNN),
Deep Neural Network (DNN), Recurrent Neural Network (RNN), and Recurrent Convolutional Neural Networks
Scientific Reports |         (2022) 12:8842  |
https://doi.org/10.1038/s41598-022-10841-5
 3
 Vol.:(0123456789)
www.nature.com/scientificreports/
 (RCNN) due to the success of the Natural Language Processing (NLP) attained by these techniques. However,
deep learning methods are not employed much in phishing detection due to the inclusive training time. Aljofey
et al.3 proposed a phishing detection approach with a character level convolutional neural network based on URL.
T
 he proposed approach was compared by using various machine and deep learning algorithms, and different
types of features such as TF-IDF characters, count vectors, and manually-crafted features. Le et al.29 provided a
URLNet method to detect phishing webpage from URL. They extract character-level and word-level features from
URL strings and employ CNN networks for training and testing. Chatterjee and  Namin30 introduced a phishing
detection technique based on deep reinforcement learning to identify phishing URLs. They used their model
on a balanced, labeled dataset of benign and phishing URLs, extracting 14 hand-crafted features from the given
URLs to train the proposed model. In recent studies, Xiao et al.31 proposed phishing website detection approach
named CNN–MHSA. CNN network is applied to extract characters features from URLs. In the meanwhile,
multi-head self-attention (MHSA) mechanism is employed to calculate the corresponding weights for the CNN
learned features. Zheng et al.32 proposed a new Highway Deep Pyramid Neural Network (HDP-CNN) which
is a deep convolutional network that integrates both character-level and word-level embedding representation
to identify whether a given URL is phishing or legitimate. Albeit the above approaches have shown valuable
performances, they might misclassify phishing websites hosted on compromised servers since the features are
extracted only from the URL of the website.
 T
 he features extracted in some previous studies are based on manual work and require additional effort
since these features need to be reset according to the dataset, which may lead to overfitting of anti-phishing
solutions. We got the motivation from the above-mentioned studies and proposed our approach. In which,
the current work extract character sequences feature from URL without manual intervention. Moreover, our
approach employs noisy data of HTML, plaintext, and hyperlinks information of the website with the benefit of
identifying new phishing websites. Table 1 presents the detailed comparison of existing machine learning based
phishing detection approaches.
 Proposed approach. Our approach extracts and analyzes different features of suspected webpages for
effective identification of large-scale phishing offenses. The main contribution of this paper is the combined uses
of these feature set. For improving the detection accuracy of phishing webpages, we have proposed eight new
features. Our proposed features determine the relationship between the URL of the webpage and the webpage
content.
 System architecture. The overall architecture of the proposed approach is divided into three phases. In
the first phase, all the essential features are extracted and HTML source code will be crawled. The second phase
applies feature vectorization to generate a particular feature vector for each webpage. The third phase identifies
if the given webpage is phishing. Figure 1 shows the system structure of the proposed approach. Details of each
phase are described as follows.
 Feature generation. The features are generated in this component. Our features are based on the URL and
HTML source code of the webpage. A Document Object Model (DOM) tree of the webpage is used to extract the
hyperlink and textual content features using a web crawler automatically. The features of our approach are cat
egorized into four groups as depicted in Table 2. In particular, features F1–F7, and F14 are new and proposed by
us; Features F8–F13, and F15 are taken from other  approaches9,11,12,24,33 but we adjusted them for better results.
Moreover, the observational method and strategy regarding the interpretation of these features are applied dif
ferently in our approach. A detailed explanation of the proposed features is provided in the feature extraction
section of this paper.
 Feature vectorization. After the features are extracted, we apply feature vectorization to generate a particu
lar feature vector for each webpage to create a labeled dataset. We integrate URL character sequences features
with textual content TF-IDF features and hyperlink information features to create feature vector required for
training the proposed approach. The hyperlink features combination outputs 13-dimensional feature vector as
FH = f3,f4,f5,...,f15 , and the URL character sequence features combination outputs 200-dimensional feature
vector as FU = �c1,c2,c3,...,c200� , we set a fixed URL length to 200. If the URL length is greater than 200, the
additional part will be ignored. Otherwise, we put a 0 in the remainder of the URL string. The setting of this
value depends on the distribution of URL lengths within our dataset. We have noticed that most of the URL
lengths are less than 200 which means that when a vector is long, it may contain useless information, in contrast
when the feature vector is too short, it may contain insufficient features. TF-IDF character level combination
outputs D-dimensional feature vector as FT = �t1,t2,t3,...,tD� where D is the size of dictionary computed from
the textual content corpus. It is observed from the experimental analysis that the size of dictionary D = 20,332
and the size increases with an increase in number of corpus. The above three feature vectors are combined to
generate final feature vector FV = FT ∪ FU ∪ FH = t1,t2,...,tD,c1,c2 ...,c200,f3,f4,f5,...,f15 that is fed as
input to machine learning algorithms to classify the website.
 Detection module. The Detection phase includes building a strong classifier by using the boosting method,
XGBoost classifier. Boosting integrates many weak and relatively accurate classifiers to build a strong and there
fore robust classifier for detecting phishing offences. Boosting also helps to combine diverse features resulting in
improved classification  performance34. Here, XGBoost classifier is employed on integrated feature sets of URL
character sequence FU , various hyperlinks information FH , login form features FL , and textual content-based
features FT to build a strong classifier for phishing detection. In the training phase, XGBoost classifier is trained
Scientific Reports |         (2022) 12:8842  |
Vol:.(1234567890)
 https://doi.org/10.1038/s41598-022-10841-5
 4
5
 Vol.:(0123456789)
 Scientific Reports |         (2022) 12:8842  | https://doi.org/10.1038/s41598-022-10841-5
 www.nature.com/scientificreports/
 using the feature vector (FU∪FH∪FL∪FT) collected from each record in the training dataset. At the testing
phase, the classifier detects whether a particular website is a malicious website or not. The detailed description
is shown in Fig. 2.
 Features extraction. Due to the limited search engine and third-party methods discussed in the literature,
we extract the particular features from the client side in our approach. We have introduced eleven hyperlink fea
tures (F3–F13), two login form features (F14 and F15), character level TF-IDF features (F2), and URL character
sequence features (F1). All these features are discussed in the following subsections.
 URL character sequence features (F1). The URL stands for Uniform Resource Locator. It is used for providing
the location of the resources on the web such as images, files, hypertext, video, etc. URL. Each URL starts with a
protocol (http, https, and ftp) used to access the resource requested. In this part, we extract character sequence
features from URL. We employ the method used  in35 to process the URL at the character level. More information
is contained at the character level. Phishers also imitate the URLs of legitimate websites by changing many unno
ticeable characters, e.g., “www.icbc.com” as “www.1cbc.com”. Character level URL processing is a solution to
the out of vocabulary problem. Character level sequences identify substantial information from specific groups
of characters that appear together which could be a symptom of phishing. In general, a URL is a string of char
Table 1.  Comparison of machine learning based phishing detection approaches.
 Approach Description Dataset Limitations
 Jain and  Gupta24
 This approach filters phishing websites at client
side based on handcrafted URL features, hyperlinks
features, and identity keywords features using
Random Forest
 A private dataset of 2141 phishing webpages and
1918 benign webpages
 It extracts manually designed URL features, which
need human effort
 Identity features are language dependent where top
key words are extracted from website
 Jain and  Gupta9
 Proposed an anti-phishing approach using logistic
regression, which relies on various hyperlink fea
tures extracted from the HTML content of webpage
 A private dataset of 1428 phishing and 1116 benign
webpages
 Limited dataset
 The feature set completely depends on the webpage
content which fails when content is replaced by
Images
 Rao and  Pais25
 Authors developed a two level filtering technique
to detect phishing sites using enhanced blacklist
and heuristic
 features
 A public dataset of 5438 benign and 4097 Phishing
webpages
 The benign sites always go through two level
filtering
 Jain and  Gupta26
 An approach to classify the websites based on two
level authentications: search engine and hyperlink
information
 A private dataset of 2000 benign and 2000 phishing
webpages
 Fails at first level when newly constructed benign
sites do not appear in top search results
 Fails when content of webpage is replaced by an
image
 Sahingoz et al.6
 Use NLP based features, word vectors, and hybrid
features, and then seven different machine learning
algorithms are used to classify the URLs
 A public dataset of 36,400 benign URLs and 37,175
phishing URLs
 Inability to handle unseen characters in URLs
 The method may fail to detect the shorter URLs
 Rao et al.13
 This technique proposes manually crafted URL
features and TF-IDF based features and with the
use of these features classifies the URLs by using
random forest classifier
 A public dataset of 85,409 benign URLs and 40,668
phishing URLs
 Extracts hand-crafted URL features, which need
human effort and additional maintenance labor
costs
 The model may fails when phishing sites hosted on
free or compromised hosting servers
 Aljofey et al.3
 A fast deep learning model based on the URL,
which uses character-level CNN, is proposed for
phishing detection
 A private dataset of 157,626 benign URLs and
161,016 phishing URLs
 It completely depends on the URL of the website
 It does not interest if the URL of the website is
alive or if there is an error
 Le et al.29
 This technique applies CNN networks to both
characters and words of the URL string for mali
cious URL detection
 A private dataset of 4,683,425 benign URLs and
9,366,850 malicious URLs
 Since the deep learning model implemented with
both word-level and character-level embedding, it
requires sufficient memory
 Xiao et al.31
 Proposed a technique named CNN–MHSA, which
combines convolutional neural network (CNN)
and multi-head self-attention (MHSA) mechanism
together to learn features in URLs and detect
phishing
 A private dataset where 45,000 are benign and
43,984 are phishing
 The URL length parameter may affect the robust
ness of the model
 Zheng et al.32
 Proposed a new Highway Hierarchical Neural
Network (HDP-CNN) to detect phishing URLs.
This method uses word-level embedding along
with character-level embedding to exhibit better
performance
 A private dataset contains 344,794 benign URLs
and 71,556 phishing URLs
 The problem of severe data imbalance is probably
causing the model to overfit on large datasets
 Rao et al.27
 A machine leaning technique that uses word
embedding algorithms to generate a feature vector
using plain text and domain text extracted from the
webpage content
 A public dataset consists of 5438 phishing websites
and 5076 benign websites with their URLs
 The technique is language dependent
 It fails when content of webpage is replaced by an
image
 Guo et al.28
 A phishing detection approach that creates hetero
geneous information networks based on domain
nodes, page resource nodes, and relationships
between hyperlinks
 A public dataset contains 29,496 phish samples and
30,649 benign samples
 The approach may exhibit poor performance when
the webpage contains a few number of hyperlinks
 Proposed approach
 A machine learning approach that consists of
a hybrid feature set including URL character
sequence, different hyperlink features, and TF-IDF
character level features from the plaintext and
noisy part of the given webpage’s HTML
 A public data set consisting of 27,280 phishing
URLs with HTML codes and 32,972 benign pages
 The plain text-based feature of a webpage is
language-based
 Need for accessing the HTML source code of
webpage
www.nature.com/scientificreports/
 Figure 1.  General architecture of the proposed approach.
 Category
 URL based features
 Textual content features
 Hyperlink information
 Login form information
 No
 F1
 F2
 F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, and F13
 Name
 Character sequences vectors
 TF-IDF vector N-gram chars
 Script_files, CSS_files, img_files, a_files, a_Null_hyperlinks, Null_hyperlinks, Total_hyperlinks,
Internal_hyperlinks, External_hyperlinks, External/Internal_hyperlinks, and Error_hyperlinks
 F14 and F15
 Total_forms and Suspicious_form
 Table 2.  Features used in the proposed approach.
 Figure 2.  Phishing detection algorithm.
 acters or words where some words have little semantic meanings. Character sequences help find this sensitive
information and improve the efficiency of phishing URL detection. During the learning task, machine learning
techniques can be applied directly using the extracted character sequence features without the expert interven
tion. The main processes of character sequences generating include: preparing the character vocabulary, creating
a tokenizer object using Keras preprocessing package (https://Keras.io) to process URLs in char level and add a
“UNK” token to the vocabulary after the max value of chars dictionary, transforming text of URLs to sequence
Scientific Reports |         (2022) 12:8842  |
Vol:.(1234567890)
 https://doi.org/10.1038/s41598-022-10841-5
 6
www.nature.com/scientificreports/
 Figure 3.  The process of generating text features.
 of tokens, and padding the sequence of URLs to ensure equal length vectors. The description of URL features
extraction is shown in Algorithm 1.
 HTML features. The webpage source code is the programming behind any webpage, or software. In case of
websites, this code can be viewed by anyone using various tools, even in the web browser itself. In this section,
we extract the textual and hyperlink features existing in the HTML source code of the webpage.
 Textual content‑based features (F2). TF-IDF stands for Term Frequency-Inverse Document Frequency. TF
IDF weight is a statistical measure that tells us the importance of a term in a corpus of  documents36. TF-IDF
vectors can be created at various levels of input tokens (words, characters, n-grams) 37. It is observed that TF
IDF technique has been implemented in many approaches to catch phish of webpages by inspecting URLs 13,
obtain the indirect associated  links38, target  website11, and validity of suspected website 39. In spite of TF-IDF
technique extracts outstanding keywords from the text content of the webpage, it has some limitations. One of
the limitations is that TF-IDF technique fails when the extracted keywords are meaningless, misspelled, skipped
or replaced with images. Since plaintext and noisy data (i.e., attribute values for div, h1, h2, body and form tags)
are extracted in our approach from the given webpage using BeautifulSoup parser, TF-IDF character level tech
nique is applied with max features as 25,000. To obtain valid textual information, extra portions (i.e., JavaScript
code, CSS code, punctuation symbols, and numbers) of the webpage are removed through regular expressions,
including Natural Language Processing packages (http://www.nltk.org/nltk_data/) such as sentence segmenta
tion, word tokenization, text lemmatization and stemming as shown in Fig. 3.
 Phishers usually mimic the textual content of the target website to trick the user. Moreover, phishers may
mistake or override some texts (i.e., title, copyright, metadata, etc.) and tags in phishing webpages to bypass
revealing the actual identification of the webpage. However, tag attributes stay the same to preserve the visual
similarity between phishing and targeted site using the same style and theme as that of the benign webpage.
T
 herefore, it is needful to extract the text features (plaintext and noisy part of HTML) of the webpage. The basic of
Scientific Reports |         (2022) 12:8842  |
https://doi.org/10.1038/s41598-022-10841-5
 7
 Vol.:(0123456789)
www.nature.com/scientificreports/
 this step is to extract the vectored representation of the text and the effective webpage content. A TF-IDF object is
employed to vectorize text of the webpage. The detailed process of the text vector generation algorithm as follows.
 Script, CSS, img, and anchor files (F3, F4, F5, and F6). External JavaScript or external Cascading Style Sheets
(CSS) files are separate files that can be accessed by creating a link within the head section of a webpage. JavaS
cript, CSS, images, etc. files may contain malicious code while loading a webpage or clicking on a specific link.
Moreover, phishing websites have fragile and unprofessional content as the number of hyperlinks referring to
a different domain name increases. We can use ![]()

 and
