from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


def kmeans_clustering(posts_file, range_min, range_max, step):
    with open(posts_file, encoding="utf-8") as inp:
        posts = inp.readlines()
    vectorizer = TfidfVectorizer(use_idf=True)
    posts_coordinates = vectorizer.fit_transform(posts)
    print("Number of features: %s" % (len(vectorizer.get_feature_names())))
    print("Number of clusters / Inertia / Diff")
    previous = 0
    for clusters_amount in range(range_min, range_max + 1, step):
        model = KMeans(
            n_clusters=clusters_amount,
            init='k-means++',
            max_iter=10,
            n_init=5,
            verbose=False
        )
        model.fit_predict(posts_coordinates)
        inertia = model.inertia_
        diff = previous - inertia
        previous = inertia
        print("%s: %s %s" % (clusters_amount, inertia, "" if clusters_amount == 1 else diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--file", help="File with posts", required=False, default="posts.txt")
    parser.add_argument("-MIN", "--min", help="Min number of clusters", type=int, required=False, default=1)
    parser.add_argument("-MAX", "--max", help="Max number of clusters", type=int, required=False, default=20)
    parser.add_argument("-STEP", "--step", help="Step", type=int, required=False, default=1)
    args = parser.parse_args()
    kmeans_clustering(args.file, args.min, args.max, args.step)
