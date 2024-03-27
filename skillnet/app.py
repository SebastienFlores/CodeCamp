from flask import Flask, render_template, request
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from flask import Flask


data = [

    {"phrase": "Ma curiosité innée me pousse à explorer de nouveaux horizons et à chercher continuellement à apprendre.", "competence": "curiosité"},
    {"phrase": "Je suis autonome dans la gestion de mes tâches et je sais prendre des initiatives sans supervision constante.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les besoins des autres et de favoriser une communication efficace.", "competence": "écoute active"},
    {"phrase": "Je suis un communicateur oral efficace, capable de transmettre mes idées de manière claire et concise.", "competence": "communication orale"},
    {"phrase": "Le respect est au cœur de mes interactions professionnelles, j'accorde de l'importance aux opinions des autres.", "competence": "respect"},
    {"phrase": "Ma flexibilité et mon adaptabilité me permettent de m'ajuster rapidement aux changements de situation.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "J'aborde chaque défi avec une attitude positive et je cherche toujours des solutions constructives.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collègues et je suis convaincu de leur capacité à accomplir leurs tâches avec succès.", "competence": "faire confiance"},
    {"phrase": "Je suis responsable de mes actions et je prends mes engagements au sérieux.", "competence": "responsabilité"},
    {"phrase": "L'intégrité guide toutes mes décisions et mes actions professionnelles.", "competence": "intégrité"},
    {"phrase": "Ma curiosité me pousse à explorer des sujets variés et à chercher des solutions innovantes.", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je prends des décisions éclairées et je suis capable de gérer des projets de bout en bout.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les besoins spécifiques de mes interlocuteurs et d'établir des relations solides.", "competence": "écoute active"},
    {"phrase": "Je suis à l'aise pour présenter des informations de manière claire et persuasive lors de réunions et de présentations.", "competence": "communication orale"},
    {"phrase": "Je traite mes collègues avec respect et je valorise leur contribution à l'équipe.", "competence": "respect"},
    {"phrase": "Ma flexibilité me permet de m'adapter aux changements de priorités et de trouver des solutions créatives aux défis.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive contribue à maintenir un environnement de travail motivant et inspirant pour tous.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collaborateurs pour accomplir leurs missions avec efficacité et professionnalisme.", "competence": "faire confiance"},
    {"phrase": "Je suis responsable de la qualité de mon travail et je m'engage à livrer des résultats de haut niveau.", "competence": "responsabilité"},
    {"phrase": "Je m'engage à agir avec intégrité dans toutes mes interactions professionnelles, en respectant les normes éthiques les plus élevées.", "competence": "intégrité"},
    {"phrase": "Ma curiosité naturelle me pousse à chercher des réponses aux questions complexes et à explorer de nouvelles idées.", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je suis capable de prendre des décisions importantes et de gérer des projets de manière efficace.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les besoins de mes collègues et de répondre de manière appropriée.", "competence": "écoute active"},
    {"phrase": "Je communique de manière claire et concise lors des réunions et des présentations pour assurer une compréhension mutuelle.", "competence": "communication orale"},
    {"phrase": "Je traite chacun avec respect et considération, en reconnaissant la valeur unique de chaque individu.", "competence": "respect"},
    {"phrase": "Ma flexibilité et mon adaptabilité me permettent de m'ajuster rapidement aux changements de situation et de trouver des solutions créatives.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive est contagieuse et contribue à créer un environnement de travail dynamique et motivant.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collègues pour accomplir leurs tâches avec compétence et je suis prêt à les soutenir.", "competence": "faire confiance"},
    {"phrase": "Je prends la responsabilité de mes actions et je cherche toujours à améliorer mes performances.", "competence": "responsabilité"},
    {"phrase": "L'intégrité est au cœur de mes interactions professionnelles, je m'engage à agir de manière honnête et éthique.", "competence": "intégrité"},
    {"phrase": "Je suis toujours curieux d'apprendre de nouvelles compétences et de développer mes connaissances dans mon domaine.", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je suis capable de travailler de manière efficace et de prendre des décisions réfléchies sans supervision constante.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les préoccupations des autres et de trouver des solutions appropriées.", "competence": "écoute active"},
    {"phrase": "Je suis un communicateur oral efficace, capable de transmettre des informations complexes de manière claire et accessible.", "competence": "communication orale"},
    {"phrase": "Je fais preuve de respect envers mes collègues et je valorise leur contribution à l'équipe.", "competence": "respect"},
    {"phrase": "Ma flexibilité me permet de m'adapter rapidement aux changements de priorités et de trouver des solutions innovantes.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive et mon enthousiasme inspirent les autres à donner le meilleur d'eux-mêmes.", "competence": "attitude positive"},
    {"phrase": "Je place ma confiance dans mes collègues et je crois en leur capacité à accomplir leurs missions avec succès.", "competence": "faire confiance"},
    {"phrase": "Je prends la responsabilité de mes actions et je suis déterminé à atteindre mes objectifs avec intégrité.", "competence": "responsabilité"},
    {"phrase": "Je m'engage à agir avec intégrité et à respecter les normes éthiques les plus élevées dans toutes mes interactions.", "competence": "intégrité"},
    {"phrase": "Ma curiosité me pousse à poser des questions et à explorer de nouvelles perspectives pour résoudre les problèmes.", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je suis capable de prendre des décisions audacieuses et de mener des projets à bien de manière indépendante.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les émotions derrière les mots et d'établir des relations de confiance.", "competence": "écoute active"},
    {"phrase": "Je suis un communicateur oral habile, capable de captiver mon public et de transmettre des informations avec conviction.", "competence": "communication orale"},
    {"phrase": "Je traite chaque individu avec respect et je suis ouvert aux idées et aux perspectives différentes.", "competence": "respect"},
    {"phrase": "Ma flexibilité et mon adaptabilité me permettent de m'ajuster facilement aux nouvelles situations et de trouver des solutions innovantes.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive et optimiste inspire les autres à rester motivés même face aux défis.", "competence": "attitude positive"},
    {"phrase": "Je choisis de faire confiance à mes collègues et je suis prêt à collaborer pour atteindre nos objectifs communs.", "competence": "faire confiance"},
    {"phrase": "Je prends la responsabilité de mes actions et je suis déterminé à assumer les conséquences de mes décisions.", "competence": "responsabilité"},
    {"phrase": "L'intégrité est au cœur de tout ce que je fais, je m'efforce toujours d'agir avec honnêteté et transparence.", "competence": "intégrité"},
    {"phrase": "Je suis constamment curieux et je cherche à élargir mes horizons en explorant de nouveaux domaines.", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je suis capable de prendre des initiatives et de résoudre les problèmes de manière créative.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de reconnaître les besoins non exprimés et d'apporter un soutien approprié.", "competence": "écoute active"},
    {"phrase": "Je suis un communicateur oral convaincant, capable de persuader et de motiver les autres.", "competence": "communication orale"},
    {"phrase": "Je fais preuve de respect envers mes collègues et je valorise leur contribution unique à l'équipe.", "competence": "respect"},
    {"phrase": "Ma flexibilité me permet de m'adapter rapidement aux changements et de trouver des solutions efficaces.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive rayonne sur les autres et crée un environnement de travail agréable et productif.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collègues pour prendre des décisions judicieuses et je les soutiens dans leurs initiatives.", "competence": "faire confiance"},
    {"phrase": "Je suis responsable de mes actions et je m'engage à atteindre les objectifs fixés avec détermination.", "competence": "responsabilité"},
    {"phrase": "L'intégrité est une valeur fondamentale pour moi, je m'efforce de toujours agir de manière éthique et juste.", "competence": "intégrité"},
    {"phrase": "Je suis à l'aise pour travailler en équipe et collaborer efficacement avec mes collègues.", "competence": "esprit d'équipe"},
    {"phrase": "J'ai toujours été quelqu'un qui apprécie collaborer avec mes collègues pour atteindre des objectifs communs.", "competence": "esprit d'équipe"},
    {"phrase": "Je suis habitué à déléguer les tâches tout en encourageant l'innovation et la prise de risque au sein de mon équipe, en permettant aux membres de tester de nouvelles idées et d'explorer des approches créatives pour résoudre les problèmes.", "competence": "esprit d'équipe"},
    {"phrase": "Je suis capable de déléguer les tâches tout en encourageant la collaboration et le partage des connaissances au sein de mon équipe, en favorisant un environnement où les membres peuvent s'entraider et apprendre les uns des autres.", "competence": "esprit d'équipe"},
    {"phrase": "Je suis habitué à déléguer les tâches tout en encourageant l'innovation et la prise de risque au sein de mon équipe, en permettant aux membres de tester de nouvelles idées et d'explorer des approches créatives pour résoudre les problèmes.", "competence": "esprit d'équipe"},
    {"phrase": "Je suis capable de déléguer les tâches tout en encourageant la collaboration et le partage des connaissances au sein de mon équipe, en favorisant un environnement où les membres peuvent s'entraider et apprendre les uns des autres.", "competence": "esprit d'équipe"},
    {"phrase": "Ma curiosité innée me pousse à explorer de nouveaux horizons et à chercher continuellement à apprendre.", "competence": "curiosité"},
    {"phrase": "Je autonome dans la gestion de mes tâches et je prendre des initiatives sans supervision constante.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les besoins des et de favoriser une communication efficace.", "competence": "écoute active"},
    {"phrase": "Je un communicateur oral efficace, capable de transmettre mes idées de manière claire et concise.", "competence": "communication efficace"},
    {"phrase": "Le respect est au cœur de mes interactions professionnelles, j'accorde de l'importance aux opinions des.", "competence": "respect"},
    {"phrase": "Ma flexibilité et mon adaptabilité me permettent de m'ajuster rapidement aux changements de situation.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "J'aborde chaque défi avec une attitude positive et je cherche toujours des solutions constructives.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collègues et je convaincu de leur capacité à accomplir leurs tâches avec succès.", "competence": "Savoir déléguer"},
    {"phrase": "L'intégrité guide toutes mes décisions et mes actions professionnelles.", "competence": "intégrité"},
    {"phrase": "Ma curiosité me pousse à explorer des sujets variés et à chercher des solutions innovantes. intéresse", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je prends des décisions éclairées et je capable de gérer des projets de bout en bout.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les besoins spécifiques de mes interlocuteurs et d'établir des relations solides.", "competence": "écoute active"},
    {"phrase": "Je à l'aise pour présenter des informations de manière claire et persuasive lors de réunions et de présentations.", "competence": "communication efficace"},
    {"phrase": "Je traite mes collègues avec respect et je valorise leur contribution à l'équipe.", "competence": "respect"},
    {"phrase": "Ma flexibilité me permet de m'adapter aux changements de priorités et de trouver des solutions créatives aux défis.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive contribue à maintenir un environnement de travail motivant et inspirant pour tous.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collaborateurs pour accomplir leurs missions avec efficacité et professionnalisme.", "competence": "Savoir déléguer"},
    {"phrase": "Je m'engage à agir avec intégrité dans toutes mes interactions professionnelles, en respectant les normes éthiques les plus élevées.", "competence": "intégrité"},
    {"phrase": "Ma curiosité naturelle me pousse à chercher des réponses aux questions complexes et à explorer de nouvelles idées. intéresse", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je capable de prendre des décisions importantes et de gérer des projets de manière efficace.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les besoins de mes collègues et de répondre de manière appropriée.", "competence": "écoute active"},
    {"phrase": "Je communique de manière claire et concise lors des réunions et des présentations pour assurer une compréhension mutuelle.", "competence": "communication efficace"},
    {"phrase": "Je traite chacun avec respect et considération, en reconnaissant la valeur unique de chaque individu.", "competence": "respect"},
    {"phrase": "Ma flexibilité et mon adaptabilité me permettent de m'ajuster rapidement aux changements de situation et de trouver des solutions créatives.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive est contagieuse et contribue à créer un environnement de travail dynamique et motivant.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collègues pour accomplir leurs tâches avec compétence et je prêt à les soutenir.", "competence": "Savoir déléguer"},
    {"phrase": "L'intégrité est au cœur de mes interactions professionnelles, je m'engage à agir de manière honnête et éthique.", "competence": "intégrité"},
    {"phrase": "Je toujours curieux d'apprendre de nouvelles compétences et de développer mes connaissances dans mon domaine.", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je capable de travailler de manière efficace et de prendre des décisions réfléchies sans supervision constante.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les préoccupations des et de trouver des solutions appropriées.", "competence": "écoute active"},
    {"phrase": "Je un communicateur oral efficace, capable de transmettre des informations complexes de manière claire et accessible.", "competence": "communication efficace"},
    {"phrase": "Je fais preuve de respect envers mes collègues et je valorise leur contribution à l'équipe.", "competence": "respect"},
    {"phrase": "Ma flexibilité me permet de m'adapter rapidement aux changements de priorités et de trouver des solutions innovantes.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive et mon enthousiasme inspirent les à donner le meilleur d'eux-mêmes.", "competence": "attitude positive"},
    {"phrase": "Je place ma confiance dans mes collègues et je crois en leur capacité à accomplir leurs missions avec succès.", "competence": "Savoir déléguer"},
    {"phrase": "Je m'engage à agir avec intégrité et à respecter les normes éthiques les plus élevées dans toutes mes interactions.", "competence": "intégrité"},
    {"phrase": "Ma curiosité me pousse à poser des questions et à explorer de nouvelles perspectives pour résoudre les problèmes. passionné", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je capable de prendre des décisions audacieuses et de mener des projets à bien de manière indépendante.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de comprendre les émotions derrière les mots et d'établir des relations de confiance.", "competence": "écoute active"},
    {"phrase": "Je un communicateur oral habile, capable de captiver mon public et de transmettre des informations avec conviction.", "competence": "communication efficace"},
    {"phrase": "Je traite chaque individu avec respect et je ouvert aux idées et aux perspectives différentes.", "competence": "respect"},
    {"phrase": "Ma flexibilité et mon adaptabilité me permettent de m'ajuster facilement aux nouvelles situations et de trouver des solutions innovantes.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive et optimiste inspire les à rester motivés même face aux défis.", "competence": "attitude positive"},
    {"phrase": "Je choisis de Savoir déléguer à mes collègues et je prêt à collaborer pour atteindre nos objectifs communs.", "competence": "Savoir déléguer"},
    {"phrase": "L'intégrité est au cœur de tout ce que je fais, je m'efforce toujours d'agir avec honnêteté et transparence.", "competence": "intégrité"},
    {"phrase": "Je constamment curieux et je cherche à élargir mes horizons en explorant de nouveaux domaines.", "competence": "curiosité"},
    {"phrase": "En tant qu'autonome, je capable de prendre des initiatives et de résoudre les problèmes de manière créative.", "competence": "autonomie"},
    {"phrase": "Mon écoute active me permet de reconnaître les besoins non exprimés et d'apporter un soutien approprié.", "competence": "écoute active"},
    {"phrase": "Je un communicateur oral convaincant, capable de persuader et de motiver les.", "competence": "communication efficace"},
    {"phrase": "Je fais preuve de respect envers mes collègues et je valorise leur contribution unique à l'équipe.", "competence": "respect"},
    {"phrase": "Ma flexibilité me permet de m'adapter rapidement aux changements et de trouver des solutions efficaces.", "competence": "flexibilité et adaptabilité"},
    {"phrase": "Mon attitude positive rayonne sur les et crée un environnement de travail agréable et productif.", "competence": "attitude positive"},
    {"phrase": "Je fais confiance à mes collègues pour prendre des décisions judicieuses et je les soutiens dans leurs initiatives.", "competence": "Savoir déléguer"},
    {"phrase": "L'intégrité est une valeur fondamentale pour moi, je m'efforce de toujours agir de manière éthique et juste.", "competence": "intégrité"},
    {"phrase": "Ma capacité à analyser les situations complexes et à proposer des solutions innovantes me permet de surmonter les défis avec efficacité.", "competence": "résolution de problèmes"},
    {"phrase": "En tant que résolveur de problèmes expérimenté, je habitué à identifier rapidement les obstacles et à élaborer des stratégies efficaces pour les résoudre. aider collègue projet", "competence": "résolution de problèmes"},
    {"phrase": "Je reconnu pour ma méthode méthodique dans la résolution de problèmes, abordant chaque situation avec calme et détermination pour trouver des solutions efficaces.", "competence": "résolution de problèmes"},
    {"phrase": "Mon approche systématique de la résolution de problèmes me permet de trouver des solutions créatives même dans les situations les plus complexes.", "competence": "résolution de problèmes"},
    {"phrase": "Je constamment à la recherche de nouvelles techniques et de nouvelles approches pour affiner mes compétences en résolution de problèmes et améliorer mes performances.", "competence": "résolution de problèmes"},
    {"phrase": "Grâce à ma capacité à penser de manière critique et à résoudre les problèmes de manière proactive, je en mesure de contribuer de manière significative à la résolution de défis organisationnels.", "competence": "résolution de problèmes"},
    {"phrase": "En tant que leader naturel, j'inspire et motive les membres de mon équipe, les guidant vers l'accomplissement de nos objectifs communs.", "competence": "leadership"},
    {"phrase": "Ma capacité à prendre des décisions difficiles et à assumer la responsabilité de mes actions fait de moi un leader fiable et efficace dans des situations complexes.", "competence": "leadership"},
    {"phrase": "Je reconnu pour ma capacité à identifier les forces individuelles de chaque membre de l'équipe et à les utiliser de manière stratégique pour atteindre nos objectifs collectifs.", "competence": "leadership"},
    {"phrase": "En tant que leader, je valorise la diversité des perspectives et je favorise un environnement inclusif où chacun se sent entendu et respecté.", "competence": "leadership"},
    {"phrase": "Ma capacité à communiquer de manière claire et persuasive me permet de mobiliser efficacement mon équipe autour d'une vision commune et de réaliser des résultats exceptionnels.", "competence": "leadership"},
    {"phrase": "Je constamment à la recherche de moyens d'améliorer mes compétences en leadership, en étant ouvert aux feedbacks et en cherchant activement des occasions de grandir et d'inspirer les.", "competence": "leadership"},
    {"phrase": "En tant que membre d'équipe dévoué, je favorise la collaboration et la synergie en encourageant la participation de chacun et en reconnaissant les contributions de tous.", "competence": "Esprit d'équipe"},
    {"phrase": "Ma capacité à travailler harmonieusement avec une variété de personnalités et à résoudre les conflits de manière constructive fait de moi un élément précieux dans tout projet d'équipe. aide", "competence": "Esprit d'équipe"},
    {"phrase": "Je reconnu pour ma flexibilité et ma volonté de prendre différents rôles au sein de l'équipe, adaptant mes compétences pour répondre aux besoins changeants du groupe.", "competence": "Esprit d'équipe"},
    {"phrase": "En tant que membre d'équipe, je déterminé à atteindre nos objectifs communs en mettant en œuvre une communication ouverte, en partageant les connaissances et en offrant mon soutien à mes collègues. aide", "competence": "Esprit d'équipe"},
    {"phrase": "Ma capacité à écouter activement les idées des et à les intégrer dans notre processus de prise de décision renforce notre cohésion d'équipe et conduit à des résultats exceptionnels.", "competence": "Esprit d'équipe"},
    {"phrase": "Je constamment à la recherche de moyens d'améliorer notre dynamique d'équipe, en encourageant l'innovation, en facilitant la résolution des conflits et en cultivant un climat de confiance et de respect mutuel.", "competence": "Esprit d'équipe"},
    {"phrase": "Je capable de prioriser efficacement les tâches, de respecter les échéances et d'optimiser ma productivité tout en maintenant un équilibre entre vie professionnelle et vie personnelle.", "competence": "gestion du temps"},
    {"phrase": "Ma capacité à planifier méthodiquement mon emploi du temps et à utiliser des outils de gestion du temps me permet de rester organisé et de maximiser mon efficacité dans toutes les facettes de ma vie.", "competence": "gestion du temps"},
    {"phrase": "Je reconnu pour ma capacité à gérer les imprévus de manière efficace, en adaptant rapidement mes plans et en restant concentré sur les objectifs prioritaires malgré les interruptions.", "competence": "gestion du temps"},
    {"phrase": "En tant que gestionnaire du temps chevronné, je capable d'identifier les activités à forte valeur ajoutée et de consacrer mes ressources temporelles de manière stratégique pour maximiser les résultats.", "competence": "gestion du temps"},
    {"phrase": "Ma pratique régulière de l'auto-évaluation me permet d'identifier les inefficacités dans ma gestion du temps et de mettre en place des ajustements pour améliorer constamment ma productivité et mon efficacité.", "competence": "gestion du temps"},
    {"phrase": "Recherche de nouvelles méthodes et techniques pour optimiser ma gestion du temps, en restant ouvert aux feedbacks et en partageant mes stratégies éprouvées avec mes collègues pour favoriser une culture de productivité au sein de l'équipe.", "competence": "gestion du temps"},
    {"phrase": "Intérêt, Désir de savoir, Fureur de découvrir, Passion de l'apprentissage, Envie d'explorer, Soif de connaissance, Intrigue, Étonnement, Éveil intellectuel, Découverte, passionné", "competence": "curiosité"}

]


phrases = [entry["phrase"] for entry in data]
competences = [entry["competence"] for entry in data]




tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(phrases)



X_train, X_test, y_train, y_test = train_test_split(X_tfidf, competences, test_size=0.2)




# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))


def predictions(model, phrase):
    return model.predict(phrase)


reponses = []
reponses_sans = []

app = Flask(__name__, static_folder='static')

@app.route("/")
def welcome_page():
    return render_template("welcome.html")

@app.route("/question1.html", methods=['POST', 'GET'])
def first_page():
    return render_template("question1.html")

@app.route("/question2.html", methods=['POST', 'GET'])
def second_page():
    html_data = request.form["enter_value"]
    # Supposons que vous avez tfidf_vectorizer et model définis ailleurs
    test = tfidf_vectorizer.transform([html_data]).toarray()
    reponse = predictions(model, test)
    reponses.append(reponse[0])
    print(reponses)
    return render_template("question2.html", html_data=reponse[0])

@app.route("/finish.html", methods=['POST'])
def finish_page():
    global reponses_sans 
    html_data = request.form["enter_value"]

    test = tfidf_vectorizer.transform([html_data]).toarray()
    reponse = predictions(model, test)

    reponses.append(reponse[0])

    return render_template("finish.html", reponse0=reponses[0], reponse1=reponses[1])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
