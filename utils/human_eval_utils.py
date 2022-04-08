import pickle
import random
from utils.self_play_train_utils import RLTrainerForGenerator
from utils.reward_utils import CoverageScorer, CoherenceScorerWoW
from utils.self_play_model_utils import BartQA, MultiBartQA
import torch
from rouge_score import rouge_scorer
import numpy as np

BASE_PATH = "/content/drive/My Drive/Talk_/"
# BASE_PATH = "../Talk_/"


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'
    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'
    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'
    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'
    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'


# Add a starting statement at the begin of the conversation
def add_pre_prompt(uttr, topic, dtype):
    if dtype == 'cd':
        return uttr
    prompt_idx = random.choice(list(range(4)))
    prompts = {
        'wow': [
            "Let's talk about %s today. ",
            "We will be discussing %s today. ",
            "The topic today will be %s. ",
            "We will talk about %s. "
        ],
        # 'cd': [
        #     "Let's focus on the following news: %s. ",
        #     "We will be talking about the breaking new %s. "
        #     "Hi! Let's talk about %s. "
        #     "I'd like to draw your attention to a breaking news: %s."
        # ],
        'papers': [
            "Let's talk about the paper '%s' ",
            "Shall we talk about the paper '%s' ",
            "Today we will be discussing '%s' ",
            "Well, let's talk about '%s' "
        ],
        'rlo': [
            "Let's talk about the paper '%s' ",
            "Shall we talk about the paper '%s' ",
            "Today we will be discussing '%s' ",
            "Well, let's talk about '%s' "
        ]

    }
    return prompts[dtype][prompt_idx] % topic + uttr


def show_response_prompt(dtype, ttype):
    prompts = {
        'wow': {
            'food': [
                "How to make this food?",
                "How do people usually eat it?",
                "Where does it originate from?",
                "What is it taste like?",
                "Give me more details."
            ],
            'city': [
                "Is there any interesting place in the city?",
                "Where is it located?",
                "What is the city's population?",
                "What is the city famous for?",
                "Tell me more about the city."
            ],
            'plane': [
                "When it its first flight?",
                'Who invented the plane?',
                "How fast can it travel?",
                "What is the plane famous for?",
                "Tell me more about ."
            ],
            'film': [
                "What story does it tell?",
                "Who stars in the film?",
                "When it is released?",
                "Which company produce the movie?",
                "Tell me more about it."
            ]
        },
        'cd': {
            'health': [
                "So, what will happen next?",
                "What will be the results?",
                "What measures are taken?",
                "What else do you know?"
            ],
            'nba': [
                "How are the players' performance?",
                "What is your opinion on the game?",
                "Who is the best player tonight?",
                "Tell me more about it."
            ],
            'tech': [
                "How do you think of the situation?",
                "So what will be the consequences?",
                "What are people's opinions?",
                "Could you tell me more?"
            ],
            'travel': [
                "What can I do in the city for fun?",
                "What is the city known for?",
                "What places are worth visiting?",
                "What else do you know?"
            ]
        },
        'papers': {
            "What did you propose?",
            "How did you do the evaluation?",
            "What is the advantage of the proposed model?",
            "What is the dataset used?",
            "How is the result?",
            'What conclusions do you have?',
        },
        'rlo': {
            '1': [
                'What are the major findings of the paper?',
                'Tell me about the ecological control conditions.',
                'Which student performs better in the experiments?',
                "What do the results suggest?",
                'Tell me more about the paper.',
                'What else do you know?',
            ],
            '2': [
                "What do you know about TNG's inclusion of a three-dimensional graphical",
                "Tell me about the methods for connecting the simulations to game dynamics",
                "Tell me about the results of the article",
                'Tell me more about the paper.',
                'What else do you know?',
            ]

        }
    }
    print('\n===Candicate responses===\n')
    if dtype in {"wow", 'cd', 'rlo'}:
        for response in prompts[dtype][ttype]:
            print('\t%s' % response)
    elif dtype == 'papers':
        for response in prompts[dtype]:
            print('\t%s' % response)
    else:
        print('Wrong dtype')
        assert False
    # return prompts[dtype][ttype]


# Anti-repetition generation
def get_wiz_say(trainer, topics, histories, documents):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    uttrs = histories[0].split(' / ')
    ids_wiz = [2 * idx + 1 for idx in range(int(len(uttrs) / 2))]
    uttrs_wiz = [uttrs[idx] for idx in ids_wiz]
    wiz_say = trainer.generate_wiz_response(topics, histories, documents, do_post_process=True)[0]
    cnt = 0
    num_no_repeat_uttrs = len(uttrs_wiz)
    while cnt < 5:
        for uttr in uttrs_wiz:
            rouge = scorer.score(uttr, wiz_say)
            if rouge['rougeL'].precision < 0.8 and rouge['rougeL'].recall < 0.8:
                num_no_repeat_uttrs -= 1
        if num_no_repeat_uttrs == 0:
            break
        wiz_say = trainer.generate_wiz_response(topics, histories, documents, do_post_process=True)[0]
    return wiz_say


def load_scorer(device):
    # coverage scorer
    scorer_cov = CoverageScorer()
    # coherence scorer
    with open(BASE_PATH + 'za/args/args_coh_wow.pkl', 'rb') as f:
        args_coh = pickle.load(f)
        args_coh.model_name_or_path = BASE_PATH + 'saved_models/coh-1.5.1/pytorch_model.bin'
    scorer_coh = CoherenceScorerWoW(args_coh, device)
    scorers = [scorer_cov, scorer_coh]
    return scorers


def load_app(device):
    with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
        args_app = pickle.load(f)
        args_app.experiment_type = 'chat_document'
        args_app.model_file_path = BASE_PATH + 'saved_models/app-1.1.1/checkpoint-29000/model.pt'
        args_app.model_name = 'facebook/bart-base'
        app = BartQA(args_app, device)
    return app


def load_wiz(name, device):
    base_args_path = BASE_PATH + 'saved_models/wiz-1.6.10.3/args.pkl'
    with open(base_args_path, 'rb') as f:
        args = pickle.load(f)
        args.out_dir = ''
        args.log_dir = ''
        args.do_post_process = True
    name2version = {
        'wow-n': None,
        'wow-v': 'wiz-1.6.9.1/step_9900',
        'wow-h': 'wiz-1.6.10.2/step_9900',
        'wow-f': 'wiz-1.6.10.3/step_9900',
        'cd-n': None,
        'cd-v': 'wiz-1.7.6.1/step_9900',
        'cd-h': 'wiz-1.7.7.2/step_9900',
        'cd-f': 'wiz-1.7.7.5/step_9900',
        'papers-n': None,
        'papers-v': 'wiz-1.8.3.1/step_9900',
        'papers-h': 'wiz-1.8.5.2/step_9900',
        'papers-f': 'wiz-1.8.5.5/step_9900'
    }
    assert name in name2version
    version = name2version[name]
    wiz_path = BASE_PATH + 'saved_models/%s/pytorch_model.bin' % version
    with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
        args_wiz = pickle.load(f)
        args_wiz.experiment_type = 'chat_document'
        args_wiz.model_file_path = args.wiz_path
        args_wiz.model_name = args.wiz_model_name
        wiz = MultiBartQA(args_wiz, device)
    if version is not None:
        wiz.generator.load_state_dict(torch.load(wiz_path))
        print('Loading %s' % wiz_path)
    else:
        print('Loading None')
    return wiz


def load_args():
    base_args_path = BASE_PATH + 'saved_models/wiz-1.6.10.3/args.pkl'
    with open(base_args_path, 'rb') as f:
        args = pickle.load(f)
        args.out_dir = ''
        args.log_dir = ''
        args.do_post_process = True
    return args


def load_trainer(args, wiz, app, scorers, accelerator):
    alphas = [1.0, 0.0]
    trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
    return trainer


def load_data(dtype, idx):
    path = BASE_PATH + 'data/human_eval/supporting_docs/'
    idx2name = {
        'wow': {
            0: 'city/Bangkok',
            1: 'city/New York City',
            2: 'city/Paris',
            3: 'city/Seattle',
            4: 'film/inception',
            5: 'film/interstellar',
            6: 'film/The Matrix',
            7: 'film/The Pursuit of Happyness',
            8: 'food/blue cheese',
            9: 'food/cheese cake',
            10: 'food/hot dry noodles',
            11: 'food/tacos',
            12: 'plane/Boeing 747',
            13: 'plane/Concorde',
            14: 'plane/f-22',
            15: 'plane/j-20',
        },
        'cd': {
            0: 'health/0',
            1: 'health/1',
            2: 'health/2',
            3: 'health/3',
            4: 'nba/4',
            5: 'nba/5',
            6: 'nba/6',
            7: 'nba/7',
            8: 'tech/8',
            9: 'tech/9',
            10: 'tech/10',
            11: 'tech/11',
            12: 'travel/12',
            13: 'travel/13',
            14: 'travel/14',
            15: 'travel/15'
        },
        'papers': {
            0: 'A Deep Reinforced Model for Abstractive Summarization',
            1: 'ALBERT',
            2: 'BART',
            3: 'BERT',
            4: 'Big Bird-Transformers for Longer Sequences',
            5: 'BottleSum-Unsupervised and Self-supervised Sentence Summarization using the Information Bottleneck Principle',
            6: 'COMET-Commonsense Transformers for Automatic Knowledge Graph Construction',
            7: 'Generating Classical Chinese Poems from Vernacular Chinese',
            8: 'Guiding Extractive Summarization with Question-Answering Rewards',
            9: 'Phrase-Based Neural Unsupervised Machine Translation',
            10: 'Reading Wikipedia to Answer Open-Domain Questions',
            11: 'RikiNet-Reading Wikipedia Pages for Natural Question Answering',
            12: 'Transformers for Image Recognition at Scale',
            13: 'Translating Embeddings for Modeling Multi-relational Data',
            14: 'Transparent Human Evaluation for Image Captioning',
            15: 'Unsupervised commonsense question answering with self-talk',
        }
    }
    name = idx2name[dtype][idx]
    if '/' in name:
        ttype, topic = name.split('/')
    else:
        ttype, topic = None, name
    with open(path + dtype + '/' + name + '.txt') as f:
        doc = f.read().replace('\n', '')
    return ttype, topic, doc


def make_human_eval_conversation(trainer, topic, document, info='wow-food'):
    topics = [topic]
    documents = [document]
    histories = ['']
    histories_real = []
    dtype, ttype = info.split('-')
    # turn 1
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    wiz_say_prompted = add_pre_prompt(wiz_say, topic, dtype)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append(Bcolors.CGREEN + "Teacher say 0" + Bcolors.ENDC +':\t%s' % wiz_say_prompted)
    print(Bcolors.CGREEN + 'Teacher say 0' + Bcolors.ENDC +':\t%s' % wiz_say_prompted)
    show_response_prompt(dtype, ttype)
    app_say = input("\nYou say 1:\t")
    histories = trainer.update_histories([app_say], histories, reverse=True)
    histories_real.append('You say 1:\t%s' % app_say)
    # turn 2
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    # wiz_say_prompted, app_response_cand = add_post_prompt(wiz_say, '', dtype1, -1)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append(Bcolors.CVIOLET + 'Teacher say 1' + Bcolors.ENDC +':\t%s' % wiz_say)
    print(Bcolors.CVIOLET + 'Teacher say 1' + Bcolors.ENDC +':\t%s' % wiz_say)
    show_response_prompt(dtype, ttype)
    app_say = input("You say 2:\t")
    histories = trainer.update_histories([app_say], histories, reverse=True)
    histories_real.append('Your say 2:\t%s' % app_say)
    # turn 3
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append(Bcolors.CYELLOW + 'Teacher say 2' + Bcolors.ENDC +':\t%s' % wiz_say)
    print('\n===Conversation history===\n')
    for uttr in histories_real:
        print(uttr)
    print('\n')


def make_human_eval_conversation_rlo(trainer, topic, document, info='rlo-1'):
    topics = [topic]
    documents = [document]
    histories = ['']
    histories_real = []
    dtype, ttype = info.split('-')
    # turn 1
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    wiz_say_prompted = add_pre_prompt(wiz_say, topic, dtype)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append(Bcolors.CGREEN + "Teacher say 0" + Bcolors.ENDC +':\t%s' % wiz_say_prompted)
    print(Bcolors.CGREEN + 'Teacher say 0' + Bcolors.ENDC + ':\t%s' % wiz_say_prompted)
    show_response_prompt(dtype, ttype)
    app_say = input("\nYou say 1:\t")
    histories = trainer.update_histories([app_say], histories, reverse=True)
    histories_real.append('You say 1:\t%s' % app_say)
    # turn 2
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append(Bcolors.CVIOLET + 'Teacher say 1' + Bcolors.ENDC + ':\t%s' % wiz_say)
    print(Bcolors.CVIOLET + 'Teacher say 1' + Bcolors.ENDC + ':\t%s' % wiz_say)
    show_response_prompt(dtype, ttype)
    app_say = input("You say 2:\t")
    histories = trainer.update_histories([app_say], histories, reverse=True)
    histories_real.append('Your say 2:\t%s' % app_say)
    # turn 3
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append(Bcolors.CBLUE + 'Teacher say 2' + Bcolors.ENDC + ':\t%s' % wiz_say)
    print(Bcolors.CBLUE + 'Teacher say 2' + Bcolors.ENDC + ':\t%s' % wiz_say)
    show_response_prompt(dtype, ttype)
    app_say = input("You say 3:\t")
    histories = trainer.update_histories([app_say], histories, reverse=True)
    histories_real.append('Your say 3:\t%s' % app_say)
    # turn 4
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append(Bcolors.CYELLOW + 'Teacher say 3' + Bcolors.ENDC +':\t%s' % wiz_say)
    print(Bcolors.CYELLOW + 'Teacher say 3' + Bcolors.ENDC + ':\t%s' % wiz_say)
    print('\n===Conversation history===\n')
    for uttr in histories_real:
        print(uttr)
    print('\n')


def get_std_input(msg, std_inputs):
    score = input(msg)
    while score not in std_inputs:
        print(Bcolors.WARNING + 'WARNING: Your input has errors, please input again' + Bcolors.ENDC)
        score = input(msg)
    return score


def collect_qa(dtype, idx):
    questions = {
        'wow': {
            0: [
                ("Bangkok is the capital and most populous city of Thailand, also known by its endonym MASKED or colloquially Krung Thep.", "Krung Thep Maha Nakhon"),
                ("The city occupies 1,568.7 square kilometres (605.7 sq mi) in the Chao Phraya River delta in central Thailand and has an estimated population of 10.539 million as of 2020, MASKED percent of the country's population.", "15.3"),
                ("Bangkok traces its roots to a small trading post during MASKED in the 15th century, which eventually grew and became the site of two capital cities, Thonburi in 1768 and Rattanakosin in 1782.", "the Ayutthaya Kingdom"),
                ("Bangkok was at the heart of the modernization of Siam, later renamed Thailand, during MASKED, as the country faced pressures from the West.", "the late-19th century"),
                ("The city, incorporated as a special administrative area under MASKED in 1972, grew rapidly during the 1960s through the 1980s", "the Bangkok Metropolitan Administration"),
            ],
            1: [
                ("Having over MASKED people in its metropolitan statistical area and 23.5 million in its combined statistical area as of 2020, New York is one of the world's most populous megacities.","20.1 million"),
                ("With a 2020 population of 8,804,190 distributed over MASKED (778.2 km2), New York City is also the most densely populated major city in the United States.","300.46 square miles"),
                ("Home to the headquarters of MASKED, New York is an important center for international diplomacy, and has sometimes been called the capital of the world.","the United Nations"),
                ("Situated on one of the world's largest natural harbors, New York City is composed of MASKED boroughs, each of which is coextensive with a respective county of the State of New York.","five"),
                ("Located at the southern tip of MASKED, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban area.","the State of New York"),
            ],
            2: [
                ("Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of MASKED.", "more than 105 square kilometres"),
                ("Since MASKED, Paris has been one of Europe's major centres of finance, diplomacy, commerce, fashion, gastronomy, science, and arts. ", " the 17th century"),
                ("The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or MASKED of the population of France as of 2017.", "about 18 percent"),
                ("The Paris Region had a GDP of MASKED in 2017.","€709 billion"),
                ("According to MASKED Worldwide Cost of Living Survey in 2018, Paris was the second most expensive city in the world, after Singapore and ahead of Zürich, Hong Kong, Oslo, and Geneva.","the Economist Intelligence Unit")
            ],
            3: [
                ("With a 2020 population of 737,015, it is the largest city in both the state of Washington and the MASKED region of North America.","Pacific Northwest"),
                (" The Seattle metropolitan area's population is 4.02 million, making it the MASKED-largest in the United States.","15th"),
                ("Its growth rate of MASKED between 2010 and 2020 makes it one of the nation's fastest-growing large cities.","21.1%、"),
                ("Seattle is situated on an isthmus between Puget Sound and MASKED.","Lake Washington"),
                ("A major gateway for trade with MASKED, Seattle is the fourth-largest port in North America in terms of container handling as of 2015.","East Asia")
            ],
            4: [
                ("The film stars MASKED as a professional thief who steals information by infiltrating the subconscious of his targets.","Leonardo DiCaprio"),
                ("Deciding he needed more experience before tackling a production of this magnitude and complexity, Nolan shelved the project and instead worked on 2005's Batman Begins, 2006's The Prestige, and MASKED in 2008.","The Dark Knight"),
                ("The treatment was revised over 6 months and was purchased by MASKED in February 2009.","Warner"),
                ("Inception's premiere was held in London on July 8, 2010; it was released in both conventional and IMAX theaters beginning on MASKED.","July 16, 2010"),
                ("Inception grossed over MASKED worldwide, becoming the fourth-highest-grossing film of 2010.","$828 million")
            ],
            5: [
                ("Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near MASKED in search of a new home for mankind.","Saturn"),
                ("MASKED, Warner Bros. Pictures, and Legendary Pictures co-financed the film.","Paramount Pictures"),
                ("Cinematographer Hoyte van Hoytema shot it on 35 mm in the MASKED anamorphic format and IMAX 70 mm.","Panavision"),
                ("The film had a worldwide gross of over $677 million (and $701 million with subsequent re-releases), making it the MASKED-highest-grossing film of 2014.","tenth"),
                ("Interstellar premiered on MASKED, in Los Angeles, California.","October 26, 2014")
            ],
            6: [
                ("The Matrix is a MASKED science fiction action film written and directed by the Wachowskis.", "1999"),
                ("It is the first installment in The Matrix film series, starring MASKED, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano.",
                "Keanu Reeves"),
                ("It depicts a dystopian future in which humanity is unknowingly trapped inside a simulated reality, the Matrix, which intelligent machines have created to distract humans while using their bodies as an MASKED.",
                "energy source"),
                ("When computer programmer Thomas Anderson, under the hacker alias \"MASKED\", uncovers the truth, he \"is drawn into a rebellion against the machines\" along with other people who have been freed from the Matrix.",
                "Neo"),
                ("The Wachowskis' approach to action scenes was influenced by MASKED and martial arts films, and the film's use of fight choreographers and wire fu techniques from Hong Kong action cinema influenced subsequent Hollywood action film productions.",
                "Japanese animation")
            ],
            7: [
                ("The Pursuit of Happyness is a 2006 American biographical drama film directed by Gabriele Muccino and starring MASKED as Chris Gardner, a homeless salesman.","Will Smith"),
                ("Smith's son MASKED co-stars, making his film debut as Gardner's son, Christopher Jr.","Jaden Smith"),
                ("The screenplay by Steven Conrad is based on the best-selling 2006 memoir of the same name written by Gardner with MASKED.","Quincy Troupe"),
                ("The film was released on December 15, 2006, by MASKED, and received moderately positive reviews, with Smith’s performance garnering universal acclaim.","Columbia Pictures"),
                ("Smith was nominated for an Oscar and MASKED.","a Golden Globe for Best Actor")
            ],
            8: [
                ("Blue cheese or bleu cheese is cheese made with cultures of the mold MASKED", "Penicillium"),
                ("Blue cheese carries a distinct smell, either from that or various specially MASKED.",
                 "cultivated bacteria"),
                ("Some blue cheeses are MASKED with spores before the curds form, and others have spores mixed in with the curds after they form.",
                "injected"),
                ("Blue cheeses are typically aged in a temperature-controlled environment such as a MASKED.", "cave"),
                ("The characteristic flavor of blue cheeses tends to be MASKED.", "sharp and salty")
                ],
            9: [
                ("The main, and thickest, layer consists of a mixture of a soft, fresh cheese, MASKED, and sugar.", "eggs"),
                ("If there is a bottom layer, it most often consists of a MASKED or base made from crushed cookies, graham crackers, pastry, or sometimes sponge cake.", "crust"),
                ("Cheesecake is usually sweetened with MASKED and may be flavored in different ways.", "sugar"),
                ("Vanilla, spices, lemon, chocolate, pumpkin, or other MASKED may be added to the main cheese layer.", "flavors"),
                ("Additional flavors and visual appeal may be added by MASKED the finished dessert with fruit, whipped cream, nuts, cookies, fruit sauce, chocolate syrup, or other ingredients.", "topping")
            ],
            10: [
                ("Hot dry noodles, known in Chinese as MASKED, also transliterated as dried and spicy noodles, is a traditional dish of Wuhan, China","reganmian"),
                ("Hot dry noodles have an MASKED history in Chinese food culture; they are unique because the noodles are not in a broth like most other Asian-style hot noodle dishes.","80-year"),
                ("They are the most significant, famous and popular MASKED food in Wuhan, often sold by street carts and restaurants in residential and business areas.","breakfast"),
                ("Breakfasts such as hot dry noodles are available from MASKED, and usually appear at Wuhan's night markets as a late-night snack.","about 5 am"),
                ("These noodles can be prepared within MASKED and are affordable, so they are a popular breakfast.","minutes")
            ],
            11: [
                ("A taco is a traditional MASKED dish consisting of a small hand-sized corn or wheat tortilla topped with a filling.","Mexican"),
                ("A taco can be made with a variety of MASKED, including beef, pork, chicken, seafood, beans, vegetables, and cheese, allowing for great versatility and variety.", "fillings"),
                ("They are often garnished with various MASKED, such as salsa, guacamole, or sour cream, and vegetables, such as lettuce, onion, tomatoes, and chiles.", "condiments"),
                ("Tacos can be contrasted with similar foods such as MASKED, which are often much larger and rolled rather than folded", "burritos"),
                ("Taco in the sense of a typical Mexican dish comprising a maize MASKED folded around food is just one of the meanings connoted by the word.","tortilla")
            ],
            12: [
                ("After introducing the MASKED in October 1958, Pan Am wanted a jet 2+1⁄2 times its size, to reduce its seat cost by 30% to democratize air travel.","707"),
                ("In 1965, Joe Sutter left the 737 development program to design the 747, the first MASKED airliner.","twin aisle"),
                ("In April 1966, Pan Am ordered 25 Boeing 747-100 aircraft and in late 1966, Pratt & Whitney agreed to develop its JT9D engine, a MASKED turbofan.","high-bypass"),
                ("On September 30, 1968, the first 747 was rolled out of the custom-built Everett Plant, the world's largest building by MASKED.","volume"),
                ("The 747 was the first airplane dubbed a \"MASKED\", the first wide-body airliner.","Jumbo Jet")
            ],
            13: [
                ("The Concorde is a British–French turbojet-powered MASKED passenger airliner that was operated from 1976 until 2003.","supersonic"),
                ("It had a maximum speed over MASKED the speed of sound, at Mach 2.04, with seating for 92 to 128 passengers.","twice"),
                ("First flown in 1969, Concorde entered service in 1976 and operated for MASKED.","27 years"),
                ("It is one of only two supersonic jetliner models to operate commercially; the other is the MASKED-built Tupolev Tu-144, which operated in the late 1970s.","Soviet"),
                ("Concorde was jointly developed and manufactured by Sud Aviation and MASKED under an Anglo-French treaty.","the British Aircraft Corporation")
            ],
            14: [
                ("The Lockheed Martin F-22 Raptor is an American single-seat, twin-engine, all-weather MASKED tactical fighter aircraft developed for the United States Air Force (USAF).","stealth"),
                ("The result of the USAF's MASKED (ATF) program, the aircraft was designed as an air superiority fighter, but also has ground attack, electronic warfare, and signals intelligence capabilities.","Advanced Tactical Fighter"),
                ("The prime contractor, Lockheed Martin, built most of the F-22's airframe and weapons systems and conducted final assembly, while MASKED provided the wings, aft fuselage, avionics integration, and training systems.","Boeing"),
                ("The fighter’s combination of stealth, aerodynamic performance, and MASKED enable unprecedented air combat capabilities.", "mission systems"),
                ("The first F-22, an EMD aircraft with tail number 4001, was unveiled at Marietta, MASKED, on 9 April 1997 and first flew on 7 September 1997.","Georgia")
            ],
            15: [
                ("The Chengdu J-20, also known as Mighty Dragon , is a single-seat, twinjet, all-weather, stealth, fighter aircraft developed by China's MASKED for the People's Liberation Army Air Force (PLAAF).","Chengdu Aerospace Corporation"),
                ("The J-20 is designed as an air superiority fighter with precision strike capability; it descends from the J-XX program of MASKED.","the 1990s"),
                ("The J-20 made its maiden flight on 11 January 2011,and was officially revealed at the 2016 MASKED.","China International Aviation & Aerospace Exhibition"),
                ("The first J-20 combat unit was formed in MASKED.","February 2018"),
                ("The J-20 is the world's third operational MASKED stealth fighter aircraft after the F-22 and F-35.","fifth-generation")
            ],
        },
        'cd': {
            0: [
                ("The US MASKED on Monday granted full approval to the Pfizer/BioNTech Covid-19 vaccine for people age 16 and older.", "Food and Drug Administration"),
                ("This is MASKED coronavirus vaccine approved by the FDA, and is expected to open the door to more vaccine mandates.", "the first"),
                ("The vaccine will be marketed as MASKED, the FDA said in its announcement on Monday.", "Comirnaty"),
                ("The Pfizer/BioNTech vaccine has been authorized for emergency use in the United States since mid-December for people age MASKED", "16 and older"),
                ("FDA says it's working as fast as possible to MASKED", "fully approve vaccines"),
            ],
            1: [
                ("The US Food and Drug Administration on Thursday authorized Merck's MASKED, molnupiravir, to treat Covid-19", "antiviral pill"),
                ("molnupiravir is for the treatment of MASKED coronavirus disease in adults with positive results of direct SARS-CoV-2 viral testing,", "mild-to-moderate"),
                ("Molnupiravir is also for who are at MASKED for progression to severe COVID-19", "high risk"),
                ("This is the second Covid-19 antiviral pill authorized for ill people to MASKED.", "take at home"),
                ("Merck has an agreement with the US government for the company to supply MASKED courses of molnupiravir upon this authorization.", "3.1 million"),
            ],
            2: [
                ("Covid-19 claimed hundreds of thousands of lives in the United States in 2020, driving a record increase in the death rate and a drop in MASKED of nearly two years", "life expectancy"),
                ("Life expectancy at birth fell MASKE years in 2020", "1.8"),
                ("This is the largest single-year decline in more than MASKED years, since World War II.", "75"),
                ("The death rate -- about 835 deaths per 100,000 people -- jumped nearly MASKED from 2019", "17%"),
            ],
            3: [
                ("The US is in a “MASKED” due to the surge in Covid-19 cases, Michael Osterholm said.", "mess right now"),
                ("It’s clear that the Omicron variant of the coronavirus is MASKED", "highly infectious"),
                ("But it is unclear how many people will get MASKED and die", "seriously sick"),
                ("The MASKED predicts more than 44,000 new Covid-19 deaths over the next four weeks.", "Centers for Disease Control and Prevention"),
            ],
            4: [
                ("The Memphis Grizzlies obliterated the Oklahoma City Thunder MASKED in Tennessee.", "152-79"),
                ("The 73-point win beats the previous 68-point record set by the MASKED", "Cleveland Cavaliers"),
                ("Led by MASKED. with 27 points, nine Grizzlies players scored double figures.", "Jaren Jackson Jr"),
                ("It will come as no great consolation to the Thunder -- reeling from an MASKED straight loss that leaves them 6-16", "eighth"),
                ("Having trailed by 78 points with just over three minutes to go, it could have been MASKED.", "worse"),
            ],
            5: [
                ("LeBron James became just the MASKED player to reach 36,000 career points in the NBA", "third"),
                ("The Los Angeles Lakers beat the MASKED 132-123 on Tuesday to end a five-game losing streak", "Houston Rockets"),
                ("Starting at center for the first time in his career, James notched MASKED points, 11 rebounds and 11 assists", "32"),
                ("The 36-year-old was aided by teammate MASKED, who notched a triple-double.", "Russell Westbrook"),
                ("With a new 'big three' of James, Westbrook and MASKED, the Lakers came into the season with championship aspirations", "Anthony Davis"),
            ],
            6: [
                ("Kevin Durant scored an NBA season high 51 points on Sunday night to help the Brooklyn Nets to a 116-104 win against the MASKED.", "Detroit Pistons"),
                ("Missing MASKED due to rest and with Kyrie Irving still absent from the team, the Nets leaned on Durant for scoring.", "James Harden"),
                ("The MASKED-year-old finished with 51 points off of 16-of-31 shooting in his 41 minutes on the floor ", "33"),
                ("His mark of 51 is the MASKED game by any player this season", "highest scoring"),
            ],
            7: [
                ("Steph Curry edged closer to becoming the NBA's all-time leader in three pointers as the Golden State Warriors beat the MASKED", "Portland Trail Blazers"),
                ("Curry hit MASKED threes en route to a team-high 22 points", "six"),
                ("An NBA Hall of Famer, Allen made MASKED three-pointers across a dazzling 18-year career", "2,973"),
                ("Curry's relentless pursuit of Allen's title has been accelerated by his and the Warriors' MASKED form this season", "scintillating"),
                ("Victory over the Trail Blazers lifted the Western Conference leaders to an NBA best MASKED.", "21-4"),
            ],
            8: [
                ("TikTok has hit one of the most exclusive milestones in the tech industry, The company said in a blog post Monday that it now has more than MASKED monthly active users around the world", "1 billion"),
                ("TikTok is the rare social media application not owned by MASKED or Google (GOOGL GOOGLE) to claim an audience of that size.", "Facebook"),
                ("The short-form video app's popularity surged as people spent more time on MASKED during the pandemic.", "their phones"),
                ("It has also both benefited from and fueled the growth of the MASKED", "creator industry"),
                ("The growth of the Chinese-owned app comes despite an effort by MASKED administration last year to shut down TikTok in the United States.", "former President Donald Trump's"),
            ],
            9: [
                ("China claimed that two SpaceX MASKED flew too close to the country's space station this year", "satellites"),
                ("forcing the station to make MASKED to avoid collision.", "evasive maneuvers"),
                ("The two encounters constituted MASKED to the life or health of astronauts aboard the China Space Station", "dangers"),
                ('China filed its complaint to the MASKED early this month.', "UN"),
                ("But the episodes didn't gain widespread attention in the country until MASKED.", "this week"),
            ],
            10: [
                ("The global MASKED is making it harder for shoppers to get their hands on some of the most sought after tech gadgets of the holiday season.", "chip shortage"),
                ("In the six weeks leading up to MASKED, some of Apple's biggest new products -- including certain iPhone 13 models, some newer iPads and AirPods -- are experiencing delays well into December", "Christmas"),
                ("Last week, some MASKED devices were showing late January shipment dates for orders", "Google Pixel 6 Pro"),
                ("by Monday, certain models were listed as MASKED on its website.", "out of stock"),
                ("Gaming consoles such as MASKED and PS5 are nearly impossible to find due to delays stemming", "Xbox"),
            ],
            11: [
                ("Hertz is betting big on MASKED, it's buying 100,000 Teslas, the largest-ever order by a single buyer.", "electric vehicles"),
                ("The purchase also represents the biggest move into EVs by a MASKED, by far.", "rental car company"),
                ("Electric vehicles are now MASKED, and we've only just begun to see rising global demand and interest", "mainstream"),
                ("The new Hertz is going to lead the way as a MASKED, starting with the largest EV rental fleet in North America and a commitment to grow our EV fleet", "mobility company"),
                ("Hertz and MASKED are having trouble getting gasoline-powered cars from traditional automakers", "its rivals"),
            ],
            12: [
                ("Hong Kong is unlike any other Asian city, its past as a cornerstone of the MASKED is obvious at every turn, from post boxes to street names.", "British Empire"),
                ("Yet this is a proudly MASKED city, with a superb food scene that's the match of anywhere else on the continent.", "Cantonese"),
                ("The views of the towering skyscrapers of Hong Kong Island from MASKED are iconic.", "Victoria Harbor"),
                ("Across the water, MASKED are grittier and reward intrepid tourists with an up-close view of the real Hong Kong.", "Kowloon's neon streets"),
                ("Beyond are some of the greenest MASKED and emptiest beaches in Asia.", "hiking spaces"),
            ],
            13: [
                ("Sydney is known around the world for its spectacular beauty and MASKED lifestyle.", "relaxed cosmopolitan"),
                ("The Australian city's waterways are its premier attraction with stunning beaches and a magnificent MASKED.", "harbor"),
                ("But another that's not to be missed is the start of the Sydney to MASKED.", "Hobart Yacht Race"),
                ("Tens of thousands of MASKED of all sizes follow the fleet out of the harbor and through the headlands.", "leisure boats"),
                ("But if you're not lucky enough to be on the water for the starting gun, you can still take part, by joining the hundreds of thousands of people who head to the MASKED in the city's eastern and northern suburbs.", "foreshore"),
            ],
            14: [
                ("Whether it's MASKED, the Houses of Parliament or Trafalgar Square, London's sights are instantly recognizable.", "Buckingham Palace"),
                ("The city's past provides ample opportunity for aimless, misty-eyed MASKED.", "strolling"),
                ('Its museums are MASKED.', "global big hitters"),
                ("Its royal parks are the ideal spot for spending MASKED", "a sunny afternoon"),
                ("Adventurous visitors can take advantage of London's superb Tube and MASKED to explore distant neighborhoods", "rail system"),
            ],
            15: [
                ("Qatar is a beguiling blend of MASKED and old-world ways.", "ultra-modern"),
                ("Its capital MASKED thrusts at the sky with the glass and steel of a 21st century megalopolis", "Doha"),
                ("For culture fans there's the stunning and expansive Museum of MASKED", "Islamic Art"),
                ("Outside the city lies the MASKED where nights under the stars and dune adventures await.", "desert"),
                ("Camel racing, beaches, falconry and Al Zubarah Fort are other highlights of the host of soccer's MASKED.", "2022 World Cup"),
            ],
        },
        'papers': {
            0: [
                ("We introduce a neural network model with a novel MASKED that attends over the input and continuously generated output separately", "intra-attention"),
                ("Models trained only with supervised learning often exhibit \"MASKED\"", "exposure bias"),
                ("When standard word prediction is combined with the MASKED the resulting summaries become more readable.", "global sequence prediction training of RL"),
                ('We evaluate this model on the MASKED.', "CNN/Daily Mail and New York Times datasets"),
                ("Human evaluation also shows that our model produces MASKED.", "higher quality summaries")
            ],
            1: [
                ('Increasing model size when pretraining natural language representations often results in MASKED.', "improved performance on downstream tasks"),
                ('At some point, further model increases become harder due to MASKED and longer training times.', "GPU/TPU memory limitations"),
                ("We present two MASKED to lower memory consumption and increase the training speed of BERT.", "parameter-reduction techniques"),
                ("Empirical evidence shows that our proposed methods lead to models that scale much better compared to MASKED.", "the original BERT"),
                ("Our best model establishes new state-of-the-art results on the MASKED, RACE, and squad benchmarks", "GLUE"),
            ],
            2: [
                ("We present BART, a denoising autoencoder for pretraining MASKED models.", "sequence-to-sequence"),
                ("BART is trained by corrupting text with an MASKED.", "arbitrary noising function"),
                ("BART is particularly effective when fine tuned for MASKED but also works well for comprehension tasks.", "text generation"),
                ("It matches the performance of MASKED with comparable training resources on GLUE and SQuAD", "RoBERTa"),
                ("BART also provides a 1.1 BLEU increase over a back-translation system for MASKED", "machine translation"),
            ],
            3: [
                ("BERT stands for MASKED.", "Bidirectional Encoder Representations from Transformers"),
                ("BERT is designed to pre-train deep bidirectional representations from MASKED.", "unlabeled text"),
                ("The pre-trained BERT model can be fine-tuned with MASKED to create state-of-the-art models for a wide range of tasks", "just one additional output layer"),
                ("BERT is conceptually MASKED and empirically powerful", "simple"),
                ("It obtains new state-of-the-art results on MASKED natural language processing tasks", "eleven"),
            ],
            4: [
                ("We propose, BigBird, a MASKED that reduces this quadratic dependency to linear.", "sparse attention mechanism"),
                ("We show that BigBird is a universal approximator of sequence functions and is MASKED.", "Turing complete"),
                ("Our theoretical analysis reveals some of the benefits of having MASKED global tokens", "O(1)"),
                ("BigBird drastically improves performance on various NLP tasks such as MASKED and summarization.", "question answering"),
                ("The proposed sparse attention can handle sequences of length up to MASKED of what was previously possible using similar hardware.", "8x"),
            ],
            5: [
                ("The principle of the MASKED is to produce a summary of information X optimized to predict some other relevant information Y.", "Information Bottleneck"),
                ("We propose a novel approach to MASKED by mapping the Information Bottleneck principle to a conditional language modelling objective", "unsupervised sentence summarization"),
                ("Using only pretrained language models with no direct supervision, our approach can efficiently perform MASKED", "extractive sentence summarization"),
                ("Building on our unsupervised extractive summarization (BottleSumEx), we then present a new approach to self-supervised MASKED", "abstractive summarization"),
                ("Empirical results demonstrate that our extractive method outperforms MASKED on multiple automatic metrics.", "other unsupervised models"),
            ],
            6: [
                ("We present the first comprehensive study on MASKED for two prevalent commonsense knowledge graphs", "automatic knowledge base construction"),
                ("Commonsense KBs only store MASKED descriptions of knowledge.", "loosely structured open-text"),
                ("An important step toward automatic commonsense completion is the development of MASKED of commonsense knowledge", "generative models"),
                ("Our investigation reveals promising results when MASKED knowledge from deep pre-trained language models is transferred to generate explicit knowledge", "implicit"),
                ("Empirical results demonstrate that COMET is able to generate MASKED that humans rate as high quality", "novel knowledge"),
            ],
            7: [
                ("Previous poem generation models only allow users to employ MASKED to interfere the meaning of generated poems", "keywords"),
                ("We propose a novel task of generating classical Chinese poems from MASKED", "vernacular"),
                ("We adapt the approach of MASKED to our task.", "unsupervised machine translation"),
                ("We explored guidelines on how to write the MASKED to generate better poems.", "input vernacular"),
                ("MASKED showed our approach can generate high-quality poems which are comparable to amateur poems.", "Human evaluation "),
            ],
            8: [
                ("A major obstacle to the development of a supervised summarizer is the lack of MASKED.", "ground-truth"),
                ("Acquiring labels by MASKED can yield inferior results", "automatically aligning human abstracts and source documents"),
                ("We describe a framework to guide a supervised, extractive summarization system with MASKED rewards.", "question-answering"),
                ("We argue that quality summaries should serve as a document surrogate to MASKED", "answer important questions"),
                ("Our results compare favorably with those reported by strong summarization baselines as evaluated by automatic metrics and MASKED.", "human assessors"),
            ],
            9: [
                ("This work investigates how to learn to translate when having access to only large MASKED in each language.", "monolingual corpora"),
                ('We propose two model variants, a MASKED and a phrase-based model.', 'neural'),
                ('Both models leverage a MASKED, the denoising effect of language models and automatic generation of parallel data.', 'careful initialization of the parameters'),
                ("These models are significantly better than methods from the literature, while being simpler and having fewer MASKED.", "hyper-parameters"),
                ("On low-resource languages like MASKED, our methods achieve even better results than semi-supervised and supervised approaches", "English-Urdu and English-Romanian"),
            ],
            10: [
                ("This paper proposes to tackle open domain question answering using MASKED as the unique knowledge source", "Wikipedia"),
                ("This task of machine reading at scale combines the challenges of MASKED with that of machine comprehension of text", "document retrieval"),
                ("Our approach combines a search component based on bigram hashing and MASKED with a multi-layer recurrent neural network model.", "TF-IDF matching"),
                ("Our experiments indicate that (1) MASKED are highly competitive with respect to existing counterparts", 'both modules'),
            ],
            11: [
                ("Reading MASKED to answer open-domain questions remains challenging in natural language understanding.", "long documents"),
                ("We introduce a new model, called RikiNet, which reads MASKED for natural question answering.", "Wikipedia pages"),
                ("RikiNet contains a MASKED and a multi-level cascaded answer predictor.", "dynamic paragraph dual-attention reader"),
                ("The reader dynamically represents the document and question by utilizing a set of MASKED.", "complementary attention mechanisms"),
                ("On the MASKED, a single RikiNet achieves 74.3 F1 and 57.9 F1 on long-answer and short-answer tasks.", "Natural Questions (NQ) dataset"),
            ],
            12: [
                ("While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to MASKED remain limited.", "computer vision"),
                ("In vision, attention is either applied in conjunction with MASKED, or used to replace certain components of convolutional networks", "convolutional networks"),
                ("We show that MASKED applied directly to sequences of image patches can perform very well on image classification tasks.", "a pure transformer"),
                ("When pre-trained on large amounts of data and transferred to multiple mid-sized or small MASKED benchmark datasets ", "image recognition"),
                ("Vision Transformer (ViT) attains excellent results compared to state-of-the-art MASKED while requiring substantially fewer computational resources to train.", "convolutional networks"),
            ],
            13: [
                ("We consider the problem of embedding MASKED of multi-relational data in low-dimensional vector spaces.", "entities and relationships"),
                ("Our objective is to propose a canonical model which is MASKED, contains a reduced number of parameters.", "easy to train"),
                ("We propose TransE, a method which models MASKED by interpreting them as translations operating on the low-dimensional embeddings of the entities.", "relationships"),
                ("Extensive experiments show that TransE significantly outperforms state-of-the-art methods in MASKED on two knowledge bases.", "link prediction"),
                ("It can be successfully trained on a MASKED set with 1M entities", 'large scale data'),
            ],
            14: [
                ("We establish a rubric-based human evaluation protocol for MASKED models.", "image captioning"),
                ("Our MASKED and their definitions are carefully developed based on machine- and human-generated captions on the MSCOCO dataset.", "scoring rubrics"),
                ("Our evaluations demonstrate several MASKED of the current evaluation practice.", "critical problems"),
                ("Human-generated captions show MASKED than machine-generated ones", "substantially higher quality"),
                ("Our rubric-based results reveal that MASKED, a recent metric that uses image features, better correlates with human judgments.", "CLIPScore"),
            ],
            15: [
                ("Current systems either rely on MASKED as the sole implicit source of world knowledge", "pre-trained language models"),
                ("We propose an unsupervised framework based on MASKED as a novel alternative to multiple-choice commonsense tasks.", "self-talk"),
                ("Inspired by inquiry-based discovery learning, our approach inquires language models with a number of MASKED", "information seeking questions"),
                ("Empirical results demonstrate that the self-talk procedure substantially improves the performance of MASKED language model baselines.", "zero-shot"),
            ],
        },
        'rlo': {
            '1': [
                ("This work focuses on gamification of shared student/system control over ______ in a linear equation tutor.",
                 "problem selection"),
                ("In a 2x2+1+1 classroom experiment with 267 ______ students, we studied the effect, on learning and enjoyment, of two ways of gamifying shared problem selection",
                 "middle school"),
                ("We also in-cluded two ______ conditions: a standard ITS and a popular algebra game, DragonBox 12+",
                 "ecological control conditions"),
                ("Of the students who had the freedom to re-practice problems, those who were not given rewards performed significantly ______ on the post-tests than their counterparts who received re-wards.",
                 "better"),
                ("Also, the students who used the tutors learned significantly more than students who used ______.",
                 "DragonBox 12+")
            ],
            '2': [
                ('StarLogo The Next Generation (TNG) enables secondary school students and teachers to model ______ through agent-based programming.',
                 "decentralized systems"),
                ("TNG's inclusion of a three-dimensional graphical environment provides the capacity to create games and simulation models with a ______.",
                 "first-person perspective"),
                ("The authors theorize that student learning of complex systems and simulations can be motivated and improved by ______ of complex systems phenomena ",
                 "transforming simulation models"),
                ("Through this transformation students interact with the model in new ways and increase their learning of both specific ______ and general processes such as inquiry, problem solving and creative thinking.",
                "content knowledge"),
                ("This article presents the results of research data from ______ of curriculum development and piloting in northern Massachusetts science classrooms",
                 "two years"),
            ],
        }
    }
    qa_pairs = questions[dtype][idx]
    judgements = []
    for (q, a) in qa_pairs:
        q = q.replace('MASKED', '______')
        print('== Question: %s\n== Answer: %s' % (q, a))
        judgement = get_std_input("\tDo you think the above question could be answered by referring to the dialogue? (Please fill in 'yes' or 'no')", {'yes', 'no'})
        print("\tYour judgement is: %s" % judgement)
        if judgement.lower() == 'yes':
            judgements.append(1)
        else:
            judgements.append(0)
    print('\tQA correct answers: %s/%s' % (sum(judgements), len(judgements)))


def collect_scores():
    print("What is the coherence score for" + Bcolors.CVIOLET + " Teacher say 1 " + Bcolors.ENDC + "to You say 1？(fill in '1',  '2', or '3')")
    coh_score1 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the coherence score for" + Bcolors.CBLUE + " Teacher say 2 " + Bcolors.ENDC + "to You say 2？(fill in '1',  '2', or '3')")
    coh_score2 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the coherence score for" + Bcolors.CYELLOW + " Teacher say 3 " + Bcolors.ENDC + "to You say 2？(fill in '1',  '2', or '3')")
    coh_score3 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CGREEN + " Teacher say 0" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score0 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CVIOLET + " Teacher say 1" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score1 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CBLUE + " Teacher say 2" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score2 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CYELLOW + " Teacher say 3" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score3 = int(get_std_input("", {'1', '2', '3'}))
    print('\tAverage Coherence score: %s\n\tAverage Readability Score: %s' % (np.mean([coh_score1, coh_score2, coh_score3]), np.mean([rdb_score0, rdb_score1, rdb_score2, rdb_score3])))


def collect_scores_rlo():
    print("What is the coherence score for" + Bcolors.CVIOLET + " Teacher say 1 " + Bcolors.ENDC + "to You say 1？(fill in '1',  '2', or '3')")
    coh_score1 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the coherence score for" + Bcolors.CBLUE + " Teacher say 2 " + Bcolors.ENDC + "to You say 2？(fill in '1',  '2', or '3')")
    coh_score2 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the coherence score for" + Bcolors.CYELLOW + " Teacher say 3 " + Bcolors.ENDC + "to You say 2？(fill in '1',  '2', or '3')")
    coh_score3 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CGREEN + " Teacher say 0" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score0 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CVIOLET + " Teacher say 1" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score1 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CBLUE + " Teacher say 2" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score2 = int(get_std_input("", {'1', '2', '3'}))
    print("What is the readability score for" + Bcolors.CYELLOW + " Teacher say 3" + Bcolors.ENDC + "？(fill in '1',  '2', or '3')")
    rdb_score3 = int(get_std_input("", {'1', '2', '3'}))
    print('\tAverage Coherence score: %s\n\tAverage Readability Score: %s' % (np.mean([coh_score1, coh_score2, coh_score3]), np.mean([rdb_score0, rdb_score1, rdb_score2, rdb_score3])))


def collect_overall():
    ova_score = int(get_std_input("What is the overall score for the conversation? (fill in '0', '1',  '2', or '3')", {'0', '1', '2', '3'}))
    print('\tOverall score: %s' % ova_score)


def collect_rank():
    print(
        "Please rank the overall performance of dialogues according to their performance in Q&A, coherence and readability scoring"
    )
    rank = input("E.g. '4>3>2>1' indicates dialogue 4 is the best dialogue, and dialogue 1 is the worst.")
    print("\tYour rank: %s" % rank)
    print('If you accidentally input wrong scores, just rerun the cell.')


def set_css():
    from IPython.display import HTML, display
    display(HTML('''
        <style>
        pre {
            white-space: pre-wrap;
        }
        </style>
        '''))



# Add a leading question at the end of the conversation
# def add_post_prompt(uttr, history, cate='food', idx=0):
#     prompts = {
#         'food': [
#             ('Do you wish to know when it is usually served?', "When do people usually eat it?"),
#             ("Would you like to know where it comes from?", "Where does it come from?"),
#             ("Do you know what it is made of?", "What is it made of?"),
#             ("You know what is special about this food?", "What is special about it?")
#         ],
#         'city': [
#             ("Do you know what the city is famous for?", 'What is the city famous for?'),
#             ("You know where it locates?", 'Where is it located?'),
#             ("Now shall we talk about the city's population?", "What is the city's population?"),
#             ("Would you like to know an interesting fact about the city?", "What is special about it?"),
#         ],
#         'plane': [
#             ("Do you know who invent this plane?", 'Which company invented the plane?'),
#             ("Would you like to know about its first flight", 'Tell me about its first flight?'),
#             ("I have a fun fact about this plane, would you like to know?", "What are other special things about it?")
#         ],
#         'film': [
#             ("Would you like to know about the story?", "What is it about?"),
#             ("Do you wish to know who made the film?", "Who made it?"),
#             ("Do you know who starred in the film?", "Who stars in the film ?"),
#             ("I have an intersting fact about the film, would you like to listen?", "What is special about it?"),
#         ]
#     }
#     added_prompt, real_question = prompts[cate][idx]
#     return uttr + ' ' + added_prompt, real_question

# def switch_lib():
#     import shutil
#     path_src = BASE_PATH + 'utils/generation_utils.py'
#     path_tgt = '/usr/local/lib/python3.7/dist-packages/transformers/generation_utils.py'
#     shutil.copy(path_src, path_tgt)

