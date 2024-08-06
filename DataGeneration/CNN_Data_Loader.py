import codecs
import logging
import os
from html.parser import HTMLParser

import pandas as pd


class ArticlePaser(HTMLParser):

    def error(self, message):
        pass

    def __init__(self):
        super().__init__()
        self.article = None
        self.query = False
        self.title = False
        self.title_tag = ""
        self.paragraph = False
        self.rating = True
        self.answer = False

    def parse_article(self, article_path):
        self.query = False
        self.title = False
        self.title_tag = ""
        self.paragraph = False
        self.rating = True
        self.answer = False
        with codecs.open(article_path, "r", "utf-8") as file:
            page = file.read()
            self.article = Article(file.name.split("/")[-1])
            self.feed(page)
        return self.article

    def handle_starttag(self, tag, attrs):
        if tag == "h1":
            if attrs:
                self.query = True
            else:
                self.title = True
        elif tag == "p":
            self.paragraph = True
            if attrs:
                a, b = attrs[0]
                if a == "rating" and b == "false":
                    self.rating = False
                if a == "answer" and b == "true":
                    self.answer = True
        elif tag == "h2" or tag == "h3":
            self.title = True
            self.title_tag = tag

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        if not data.strip():
            return
        if self.query:
            self.article.set_query(data)
            self.query = False
        elif self.title:
            self.article.add_title(title_text=data, title_tag=self.title_tag)
            self.title_tag = ""
            self.title = False
        elif self.paragraph:
            self.article.add_paragraph(text=data, rating=self.rating, answer=self.answer)
            self.rating = True
            self.paragraph = True
            self.answer = False


class Article(object):
    """Article Representation"""

    def __init__(self, name):
        self._paragraphs = []
        self._content = []
        self._query = ""
        self._title = ""
        self._file_name = name
        self._para_count = 0
        self._answer_par = 0

    def set_query(self, query):
        self._query = query

    def add_paragraph(self, text, rating, answer):
        if answer:
            self._answer_par = self._para_count
        word_count = len(text.bhattacharya_split(" "))
        self._paragraphs.append(
            {"type": "paragraph", "num": self._para_count, "answer": answer, "word_count": word_count, "text": text})
        self._content.append(
            {"type": "paragraph", "num": self._para_count, "answer": answer, "word_count": word_count, "text": text})
        self._para_count += 1

    def add_title(self, title_text, title_tag):
        if self._title == "":
            self._title = title_text
        else:
            self._content.append({"type": title_tag, "answer": False, "text": title_text})

    @property
    def query(self):
        return self._query

    @property
    def content(self):
        return self._content

    @property
    def num_answer(self):
        return self._answer_par

    @property
    def num_paragraph(self):
        return self._para_count

    @property
    def paragraphs(self):
        return self._paragraphs

    @property
    def title(self):
        return self._title

    @property
    def file_name(self):
        return self._file_name

    @property
    def article_id(self):
        return self._file_name[:-5]


class SciBot_DataLoader:
    _EXCLUDE_USER_LIST = []  # ["A02", "A05"]  # exclude A02 for g-REL only
    _INCLUDE_USER_LIST = []  # if emtpy every user besides excluded will be analysed, "A01", "B01"
    _INCLUDE_DATA_SOURCE = []  # "g_rel", "GoogleNQ"
    _STUDY_TYPE = ["main"]  # "main", "train"
    _TASK_TYPES = []
    _GREL_PARAGRAPH_BOTTOM = {
        'q103-1': 1296, 'q118-1': 1243, 'q122-2': 1455, 'q097-2': 1361, 'q094-2': 1349, 'q085-2': 1296,
        'q076-1': 1422, 'q075-1': 1190, 'q134-3': 1349, 'q128-1': 1316, 'q116-1': 1283, 'q088-1': 1316
    }

    google_nq_reading = {}
    google_nq_rating = {}
    google_nq_articles = {}
    grel_reading = {}
    grel_rating = {}
    grel_articles = {}

    @property
    def num_participants(self):
        return len(self.grel_reading.keys())

    @property
    def num_scanpaths_nq_reading(self):
        return self.num_scanpaths("nq", type="reading")

    @property
    def num_scanpaths_grel_reading(self):
        return self.num_scanpaths("grel", type="reading")

    def num_scanpaths(self, corpus="grel", type="reading"):
        d = None
        if corpus == "grel":
            d = self.grel_reading if type == "reading" else self.grel_rating
        elif corpus == "nq":
            d = self.google_nq_reading if type == "reading" else self.google_nq_rating

        return sum([len(p.keys()) for k, p in d.items()])

    @property
    def num_paragraphs_nq_reading(self):
        return self.num_paragraphs("nq", type="reading")

    @property
    def num_paragraphs_grel_reading(self):
        return self.num_paragraphs("grel", type="reading")

    def num_paragraphs(self, corpus="grel", type="reading"):
        d = None
        if corpus == "grel":
            d = self.grel_reading if type == "reading" else self.grel_rating
        elif corpus == "nq":
            d = self.google_nq_reading if type == "reading" else self.google_nq_rating

        return sum([sum([s["num_paragraphs"] for _, s in p.items()]) for _, p in d.items()])

    def __init__(self, data_dir: str, googleNQ=True, gREL=True, gaze_data=True,
                 training_data=False, include_users=None, exclude_users=None,
                 reading_task=True, rating_task=False, article_dir="", ):
        """
        @param data_dir: The path to the data directory that contains the SciBot data recordings.
        @param googleNQ: Specifies whether data for GoogleNQ stimuli is loaded
        @param gREL: Specifies whether data for gREL stimuli is loaded
        @param gaze_data: Specifies whether the gaze data is loaded (less memory load, if ratings are required only)
        @param training_data: If set to True, the training data is loaded instead of data from the main experiment
        @param include_users: A list of users for which the data shall be loaded.
        If set to None, all but the excluded users are loaded.
        @param exclude_users: A list of users for which no data shall be loaded
        @param reading_task: Data from the reading tasks are loaded.
        @param rating_task: Data from the rating tasks are loaded.
        """
        # set configuration parameters
        self.data_dir = data_dir
        self.gaze_data = gaze_data
        if googleNQ:
            self._INCLUDE_DATA_SOURCE.append("GoogleNQ")
        if gREL:
            self._INCLUDE_DATA_SOURCE.append("g-rel")
        if training_data:
            self._STUDY_TYPE = ["train"]
        if include_users is not None:
            assert isinstance(include_users, list)
            self._INCLUDE_USER_LIST = include_users
        if exclude_users is not None:
            assert isinstance(exclude_users, list)
            self._EXCLUDE_USER_LIST = exclude_users
        if reading_task:
            self._TASK_TYPES.append("Reading")
        if rating_task:
            self._TASK_TYPES.append("Rating")

        # load data from the scibot study
        self._load_data()

        if article_dir:
            self.article_dir = article_dir
            self._load_articles()

    def _load_data(self):
        """
        initialize and load the study data
        """
        for user_data in os.scandir(self.data_dir):
            if user_data.is_file():
                logging.warning("Data directory contains an unknown file!")
            elif user_data.name not in self._EXCLUDE_USER_LIST and (
                    not self._INCLUDE_USER_LIST or user_data.name in self._INCLUDE_USER_LIST):
                logging.debug(user_data.name)
                self._load_user_data(user_data)

    def _load_user_data(self, user_data):
        if "g-rel" in self._INCLUDE_DATA_SOURCE:
            self._load_gREL(user_data)
        if "GoogleNQ" in self._INCLUDE_DATA_SOURCE:
            self._load_googleNQ(user_data)

    def _init_user_rating(self, path):
        with open(os.path.join(path, "User_Rating")) as rating_file:
            rating = []
            for line in rating_file:
                if line.strip():
                    rating.append([r == "True" for r in line.strip().split("|")])
            return rating

    def _load_gREL(self, user_data):
        """
           load g-rel data
        """
        paths = [os.path.join(user_data.path, "g-rel", study_type) for study_type in self._STUDY_TYPE]
        self.grel_reading[user_data.name] = {}
        self.grel_rating[user_data.name] = {}
        for current_path in paths:
            answers = self._init_user_rating(current_path)
            # logging.debug(current_path)
            for log_file in os.scandir(current_path):
                if log_file.is_file() and log_file.name.endswith(".csv"):
                    try:
                        index, corpus, doc, sys_relevance, file_ext, task_type = \
                            log_file.name[:-4].replace(".", "_").split("_")
                    except ValueError:
                        raise ValueError("Wrong filename format!")

                    if task_type not in self._TASK_TYPES:
                        continue

                    obj = {
                        "index": int(index), "doc": doc, "corpus": corpus, "file": log_file.name,
                        "perceived_relevance": answers[int(index)], "num_paragraphs": 1,
                        "system_relevance": [sys_relevance == "r"], "g-rel_relevance": sys_relevance
                    }
                    if self.gaze_data:
                        df = pd.read_csv(log_file.path, delimiter="|", encoding='utf-8', float_precision='round_trip',
                                         na_values="None")
                        self._grel_paragraph_reassignment(obj["doc"], df)
                        df.insert(loc=3, column="gaze_y_abs", value=df["gaze_y"])  # no scrolling for g-REL => y=abs(y)
                        df.drop(columns=["fixation_x", "fixation_y"], inplace=True)  # fixation points are recomputed
                        obj["dataframe"] = df
                    if task_type == "Rating":
                        self.grel_rating[user_data.name][("_".join([corpus, doc]))] = obj
                    else:
                        self.grel_reading[user_data.name][("_".join([corpus, doc]))] = obj

    def _grel_paragraph_reassignment(self, document, df):
        """Checks whether gaze/fixation points lie within a g-REL paragraph. Changes paragraph assignment inplace."""
        x1, x2 = 749, 1773
        y1, y2 = 1440 - 0, 1440 - self._GREL_PARAGRAPH_BOTTOM[document]  # gaze and fixation origin is lower-left corner

        # perform reassignment for paragraph_id
        paragraph_mask = (df["gaze_x"] > x1) & (df["gaze_x"] < x2) & (df["gaze_y"] < y1) & (df["gaze_y"] > y2)
        df["paragraph_id"] = -2
        df.loc[paragraph_mask, "paragraph_id"] = 0

        # perform reassignment for fixation_paragraph_id
        fixation_paragraph_mask = df["fixation_id"].notna() & \
                                  (df["fixation_x"] > x1) & (df["fixation_x"] < x2) & \
                                  (df["fixation_y"] < y1) & (df["fixation_y"] > y2)
        df.loc[df["fixation_id"].notna(), "fixation_paragraph_id"] = -2
        df.loc[fixation_paragraph_mask, "fixation_paragraph_id"] = 0

    def _load_googleNQ(self, user_data):
        """
            load GoogleNQ data
        """
        paths = [os.path.join(user_data.path, "GoogleNQ", study_type) for study_type in self._STUDY_TYPE]
        self.google_nq_reading[user_data.name] = {}
        self.google_nq_rating[user_data.name] = {}
        for current_path in paths:
            # logging.info(current_path)
            answers = self._init_user_rating(current_path)
            for log_file in os.scandir(current_path):
                if log_file.is_file() and log_file.name.endswith(".csv"):
                    try:
                        index, corpus, par_size, answ_par, name, file_ext, task_type = \
                            log_file.name[:-4].replace(".", "_").split("_")
                    except ValueError:
                        raise ValueError("Wrong filename format!")

                    if task_type not in self._TASK_TYPES:
                        continue

                    num_paragraphs = int(par_size[:-1])
                    answer_paragraph_id = int(answ_par[1:])
                    sys_relevance = [False] * num_paragraphs
                    sys_relevance[answer_paragraph_id] = True

                    obj = {
                        "index": int(index), "doc": "_".join([par_size, answ_par, name]), "corpus": corpus,
                        "file": log_file.name, "num_paragraphs": num_paragraphs,
                        "answer_paragraph_id": answer_paragraph_id, "perceived_relevance": answers[int(index)],
                        "system_relevance": sys_relevance
                    }
                    if self.gaze_data:
                        df = pd.read_csv(log_file.path, delimiter="|", encoding='utf-8', float_precision='round_trip',
                                         na_values="None")
                        df.insert(loc=3, column="gaze_y_abs", value=df["abs_y"])
                        df.drop(columns=["fixation_x", "fixation_y", "abs_y"], inplace=True)  # fixations are recomputed
                        obj["dataframe"] = df

                    if task_type == "Rating":
                        self.google_nq_rating[user_data.name]["_".join([corpus, par_size, answ_par, name])] = obj
                    else:
                        self.google_nq_reading[user_data.name]["_".join([corpus, par_size, answ_par, name])] = obj

    def _load_articles(self):
        article_parser = ArticlePaser()
        for data_source in self._INCLUDE_DATA_SOURCE:
            for study_type in self._STUDY_TYPE:
                folder_dir = os.path.join(self.article_dir, data_source, study_type)
                for file in os.listdir(folder_dir):
                    article_id = file[:-5]
                    if data_source == "GoogleNQ":
                        self.google_nq_articles[article_id] = \
                            article_parser.parse_article(os.path.join(folder_dir, file))
                    else:
                        self.grel_articles[article_id] = \
                            article_parser.parse_article(os.path.join(folder_dir, file))
