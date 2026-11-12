
import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------
# 1. SET UP PROJECT PATHS
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found!")

# --------------------------------
# 2. CONNECT TO GEMINI
# --------------------------------

client = genai.Client(api_key=API_KEY)

# --------------------------------
# 3. CREATE CHROMADB DATABASE
# --------------------------------

DB_PATH = os.path.join(BASE_DIR, "rag", "vector_store")

chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = chroma_client.get_or_create_collection(
    name="footiq_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# --------------------------------
# 4. FOOTBALL KNOWLEDGE DOCUMENTS
# --------------------------------

documents = [
    {
        "id": "football_1",
        "topic": "High Pressing",
        "text": """
        High pressing is a football strategy where players
        put pressure on opponents high up the pitch to regain
        possession close to the opponent's goal. It can force
        mistakes and create attacking opportunities.
        """
    },
    {
        "id": "football_2",
        "topic": "4-3-3 Formation",
        "text": """
        The 4-3-3 formation consists of four defenders,
        three midfielders and three forwards. It provides
        width in attack and allows teams to control the
        midfield while maintaining attacking options.
        """
    },
    {
        "id": "football_3",
        "topic": "Counter Attacking",
        "text": """
        Counter-attacking is a strategy where a team quickly
        attacks immediately after winning possession.
        Players exploit spaces left by opponents who have
        moved forward to attack.
        """
    },
    {
        "id": "football_4",
        "topic": "Possession Football",
        "text": """
        Possession-based football focuses on maintaining
        control of the ball through passing and movement.
        Teams use possession to control the tempo, create
        spaces and build attacking opportunities.
        """
    },
    {
        "id": "football_5",
        "topic": "Elo Rating",
        "text": """
        Elo rating is a numerical rating system used to
        estimate the relative strength of teams. A team
        gains rating points after performing better than
        expected and loses points after performing worse
        than expected.
        """
    },
    {
        "id": "football_6",
        "topic": "Expected Goals",
        "text": """
        Expected Goals, commonly called xG, measures the
        quality of goal-scoring opportunities. Each shot
        receives a probability of becoming a goal based on
        factors such as shot location and situation.
        """
    },
    {
        "id": "football_7",
        "topic": "Home Advantage",
        "text": """
        Home advantage refers to the potential benefits
        teams receive when playing at their home stadium.
        Familiar surroundings, crowd support and travel
        differences can influence match performance.
        """
    },
    {
        "id": "football_8",
        "topic": "Football Match Outcomes",
        "text": """
        Football match outcomes are commonly classified
        into three categories: home win, draw and away win.
        Match prediction models use historical results and
        performance features to estimate the probability
        of each outcome.
        """
    },
    {
        "id": "football_9",
        "topic": "Offside Rule",
        "text": """
        A player is in an offside position if:
        1. They are in the opponents' half of the pitch.
        2. Any part of their head, body or feet is nearer
        to the opponents' goal line than the ball.
        3. Any part of their head, body or feet is also
        nearer to the opponents' goal line than the
        second-last opponent.
        This position is judged at the moment a teammate
        plays or touches the ball.
        Being in an offside position is not an offence
        by itself. An offence occurs only when the player
        becomes involved in active play, such as interfering
        with play, interfering with an opponent, or gaining
        an advantage.
        """
   },
   {
       "id": "football_10",
       "topic": "Defending",
       "text": """
       Defending is the process of preventing the opposing team
       from scoring goals and regaining possession of the football.

       The main objective is to protect the team's goal, restrict
       the opponent's attacking opportunities and maintain defensive
       organization.

       Important defensive principles include:

       1. Positioning: Players maintain suitable positions to block
          passing lanes and protect dangerous areas.

       2. Marking: Defenders track opposing players to prevent them
          from receiving the ball freely.

       3. Tackling: Players attempt to legally win possession of
       the ball from an opponent.

       4. Interception: Players read the opponent's passes and
          intercept the ball.

       5. Covering: Teammates provide support when another defender
          is beaten or moves out of position.

       6. Defensive Shape: Players maintain an organized formation
          to reduce spaces between defensive lines.

       Common defensive approaches include low block, high pressing,
       man marking, zonal marking and counter-pressing.
      """
    },
    {
        "id": "football_11",
        "topic": "Low Block",
        "text": """
        A low block is a defensive football strategy in which a team
        positions most of its players deep in its own half.

        The objective is to reduce space near the team's goal,
        restrict passing opportunities and make it difficult for
        opponents to create scoring chances.

        Teams using a low block generally allow opponents more
        possession while prioritizing defensive organization.

        A low block can be effective against teams that dominate
        possession, but it requires good positioning, communication
        and discipline.
        """
    },
    {
        "id": "football_12",
        "topic": "Man Marking",
        "text": """
        Man marking is a defensive strategy in which a player is
        assigned to closely track a specific opponent.

        The objective is to limit that opponent's freedom to receive
        passes, turn with the ball or create attacking opportunities.

        Man marking can be used across the pitch or in specific areas.

        Its advantages include close defensive pressure and reducing
        the influence of important attacking players.

        Its disadvantages include creating spaces when players move
        away from their assigned opponents.
        """
    },
    {
        "id": "football_13",
        "topic": "Zonal Marking",
        "text": """
        Zonal marking is a defensive strategy in which players are
        responsible for protecting specific areas of the pitch
        instead of following individual opponents everywhere.

        Players maintain their defensive positions and challenge
        opponents who enter their assigned zones.

        Zonal marking helps maintain team structure and defensive
        organization, particularly during set pieces.
        Its advantages include reducing spaces between defenders
        and allowing players to cover multiple opponents."""
    },
    {
        "id": "football_14",
        "topic": "Counter Pressing",
        "text": """
        Counter pressing, also known as counter-pressing or
        gegenpressing, is a football defensive tactic where a team
        immediately tries to win back possession after losing the ball.

        It is also called pressing immediately after losing possession
        or winning the ball back quickly after a turnover.

        Instead of retreating into a defensive shape, nearby players
        quickly pressure the opponent who has just gained possession.
        They attempt to block passing options, force mistakes and
        recover the ball before the opponent can organize an attack.

        The main objectives of counter pressing are:
        1. Regain possession immediately after losing the ball.
        2. Prevent the opponent from launching a counterattack.
        3. Exploit the opponent's disorganized defensive shape.
        4. Create quick attacking opportunities.

        Counter pressing requires teamwork, quick reactions,
        coordinated movement, good positioning and high physical
        intensity.

        Counter pressing is different from ordinary high pressing.
        High pressing involves applying pressure high up the pitch,
        whereas counter pressing specifically happens immediately
        after a team loses possession.
        """

    },
    {    
        "id": "football_15",
        "topic": "Offensive Football",
        "text": """
        Offensive football refers to the strategies and movements
        a team uses to create scoring opportunities.

        Important attacking principles include creating space,
        passing combinations, movement without the ball,
        width, depth and penetration.

        Common attacking approaches include possession-based play,
        counterattacking, direct football and positional play.

        The objective is to progress the ball toward the opponent's
        goal and create opportunities to score.
        """
    }

]

# --------------------------------
# 5. GENERATE GEMINI EMBEDDINGS
# --------------------------------

ids = []
texts = []
embeddings = []
metadatas = []

print("Generating football knowledge embeddings...")

for document in documents:

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=document["text"],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    embedding = response.embeddings[0].values

    ids.append(document["id"])
    texts.append(document["text"])
    embeddings.append(embedding)
    metadatas.append({
        "topic": document["topic"]
    })

    print("Embedded:", document["topic"])

# --------------------------------
# 6. STORE IN CHROMADB
# --------------------------------

collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas
)

print("\nKnowledge base created successfully!")
print("Total documents:", collection.count())
print("Database location:", DB_PATH)
