class DatasetEnum:
    AMBIQT = "ambiqt"
    AMBROSIA = "ambrosia"
    BIRD = "bird"
    SPIDER = "spider"
    CLAMBSQL = "clambsql"

class MappingEnum:
    ALL = "all"
    QUERY = "query"
    MATCH = "match"

class SelectionEnum:

    HUMAN = "human"
    VOTING = "voting"
    ALL = "all"

    class HumanSelector:
        INPUT = "input"
        APP = "app"

    class VotingSelector:
        LIKELIHOOD = "likelihood"
        LLM = "llm"
        GOLD = "gold"
        RANDOM = "random"

APP_HOST = '0.0.0.0'
APP_PORT = 5000