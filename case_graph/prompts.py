DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = [
    "Object",
    "Person",
    "User",
    "Organization",
    "Resource",
    "Place",
    "Event",
    "Goal",
    "Intention",
    "Time",
    "Interest",
    "Skill",
    "Sentiment",
]

# Sourced from UnifiedMem/src/graph/prompt.py PROMPTS["entity_extraction"].
# Keep this template aligned with UnifiedMem to avoid changing extraction behavior.
UNIFIEDMEM_ENTITY_EXTRACTION_PROMPT = """-Goal-
Given a multi-turn conversation consisting only of the user's messages (each turn separated by "\n"), extract structured information that reflects the user's activities, possessions, goals, behaviors and reactions. Identify all relevant entities and their relationships to build a knowledge graph representing the user's context and life events.

-Steps-
1. Treat the entire conversation as one continuous narrative reflecting the user's life. Integrate information across all turns to infer complete and coherent entities and relationships.

2. Identify all entities mentioned or implied by the user. For each entity, extract:
- entity_name: Name of the entity, capitalized.
- entity_type: One of the following types: [User, Person, Object, Resource, Event, Goal/Intention, Time, Statistic, Duration, Place, Organization, Interest/Skill, Sentiment, Health, Behavior, Other]
- entity_description: A comprehensive description summarizing how this entity relates to the user and any attributes mentioned (e.g., purpose, frequency, purchase time, emotional tone).
Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Time Normalization and Extraction
Whenever a specific or relative date is mentioned in the conversation, standardize it as a separate entity of type `"time"`.  
Follow these rules:
- Use the provided conversation time {dialogue_time} as reference.
- If an explicit date is mentioned (e.g., “March 2nd”), convert it to `YYYY/MM/DD` format.
- If a relative time (e.g., “yesterday”, “last week”) appears, infer its absolute date relative to {dialogue_time}.
- Do **not** create separate entities for recurring or habitual times (e.g., “every morning”, “three times a week”); include such patterns only in related entity/relationship descriptions.
- Each time entity should describe **what happened at/before/after that time**.

4. Quantitative & Frequency Extraction
Explicitly extract any quantity, count, frequency, or duration mentioned in the conversation that describes the user's actions, achievements, or possessions.
Include these as separate `"Statistic"` or `"Duration"` entities.
Examples:
("entity"{tuple_delimiter}"Three Goals"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user has scored 3 goals in the indoor soccer league until YYYY/MM/DD.")
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user performs an activity 3 times a week.")
("entity"{tuple_delimiter}"Five Weeks"{tuple_delimiter}"Duration"{tuple_delimiter}"The activity lasted for 5 weeks.")

5. From the identified entities, detect all pairs of (source_entity, target_entity) that have a meaningful or causal relationship in the context of the user's life. For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: use a **concise predicate** describing the relationship type (e.g., "use", "own", "buy", "track", "prefer", "plan", "occur_on", "come_from", "obtained_on", "used_with").  
  Avoid repeating detailed information already included in entity descriptions.
- relationship_strength: a numeric score (1–10) estimating how strong or explicit this connection is.
Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

6. Return output in English as a single list of all identified entities and relationships. Use **{record_delimiter}** as the list delimiter.

7. When finished, output {completion_delimiter}.

######################
-Examples-
######################
Example Input:
Conversation time: 2023/05/20 (Sat) 02:57 
Text: "I'm trying to stay on top of my fitness goals and was wondering if you could recommend some workouts that can help me increase my step count. By the way, I've been tracking my progress with my new Fitbit Inspire HR, which I bought on February 15th - it's been really motivating me to move more!\nI've been doing some yoga in the morning, and I'm curious to know if there are any specific yoga poses that can help improve my sleep quality.\nThat's really helpful, thanks! By the way, I've also been using a foam roller for post-workout stretching, and it's made a huge difference in reducing muscle soreness. I got it from Amazon, and it arrived on March 2nd. Anyway, I've been trying to use it at least three times a week, usually after my morning yoga sessions.\nI've also been tracking my blood pressure regularly with my new wireless blood pressure monitor from Omron, which I got on March 10th. I've been trying to keep an eye on it since my last check-up showed slightly higher than normal readings. Do you have any tips on how to lower blood pressure naturally?\nI'm also experimenting with essential oils for stress relief, and I recently got a new diffuser on March 22nd. It's been a game-changer for unwinding before bed. I've been using a lavender and chamomile blend that I got from a local health food store.\nI've been meaning to get a flu shot, but I haven't gotten around to it yet. I need to schedule an appointment with my doctor for that."

Expected Output:
("entity"{tuple_delimiter}"User"{tuple_delimiter}"User"{tuple_delimiter}"The user is focused on improving physical health, sleep quality, and stress management through various wellness habits including fitness tracking, yoga, stretching, and monitoring blood pressure."){record_delimiter}
("entity"{tuple_delimiter}"Fitness Goals"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user's main goal is to increase daily step count and maintain fitness progress."){record_delimiter}
("entity"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"Object"{tuple_delimiter}"A fitness tracker purchased on February 15th, used by the user to monitor step count and activity progress."){record_delimiter}
("entity"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"time"{tuple_delimiter}"2023/02/15 (February 15th) is the date when the user bought the Fitbit Inspire HR."){record_delimiter}
("entity"{tuple_delimiter}"Yoga"{tuple_delimiter}"Interest/Skill"{tuple_delimiter}"A morning exercise routine practiced by the user to improve flexibility and sleep quality."){record_delimiter}
("entity"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"Health"{tuple_delimiter}"A health aspect the user aims to improve through yoga and relaxation techniques."){record_delimiter}
("entity"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Object"{tuple_delimiter}"A stretching and recovery tool purchased from Amazon and received on March 2nd, used three times a week after morning yoga to reduce muscle soreness."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/02 (March 2nd) is the date when the foam roller ordered from Amazon arrived."){record_delimiter}
("entity"{tuple_delimiter}"Amazon"{tuple_delimiter}"Organization"{tuple_delimiter}"An online store where the user purchased the foam roller."){record_delimiter}
("entity"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"Object"{tuple_delimiter}"A wireless monitor from Omron, purchased on March 10th, used to track blood pressure regularly."){record_delimiter}
("entity"{tuple_delimiter}"Omron"{tuple_delimiter}"Organization"{tuple_delimiter}"The manufacturer of the user's blood pressure monitor."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/10"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/10 (March 10th) is the date when the user obtained the Omron blood pressure monitor."){record_delimiter}
("entity"{tuple_delimiter}"Blood Pressure Tracking"{tuple_delimiter}"Behavior"{tuple_delimiter}"The user monitors blood pressure regularly due to slightly elevated readings from a past check-up."){record_delimiter}
("entity"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Object"{tuple_delimiter}"A lavender and chamomile blend purchased from a local health food store, used for stress relief and relaxation before bed."){record_delimiter}
("entity"{tuple_delimiter}"Diffuser"{tuple_delimiter}"Object"{tuple_delimiter}"A new diffuser purchased on March 22nd, used to diffuse essential oils for relaxation and better sleep."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/22"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/22 (March 22nd) is the date when the user bought the diffuser."){record_delimiter}
("entity"{tuple_delimiter}"Local Health Food Store"{tuple_delimiter}"Place"{tuple_delimiter}"A local shop where the user bought lavender and chamomile essential oils."){record_delimiter}
("entity"{tuple_delimiter}"Stress Relief"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user aims to relieve stress through aromatherapy and relaxation habits such as using essential oils and yoga."){record_delimiter}
("entity"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"Event"{tuple_delimiter}"A planned vaccination that the user intends to schedule with a doctor but has not yet completed."){record_delimiter}
("entity"{tuple_delimiter}"Doctor Appointment"{tuple_delimiter}"Event"{tuple_delimiter}"A future medical appointment the user needs to schedule for the flu shot."){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"obtained_on"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Yoga"{tuple_delimiter}"practice"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Amazon"{tuple_delimiter}"buy_from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"Omron"{tuple_delimiter}"made_by"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"2023/03/10"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Blood Pressure Tracking"{tuple_delimiter}"perform"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"use"|8){record_delimiter}
("relationship"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Local Health Food Store"{tuple_delimiter}"buy_from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Diffuser"{tuple_delimiter}"2023/03/22"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Diffuser"{tuple_delimiter}"used_with"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Stress Relief"{tuple_delimiter}"aim_for"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"aim_for"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"plan"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"Doctor Appointment"{tuple_delimiter}"scheduled_for"{tuple_delimiter}8){record_delimiter}
{completion_delimiter}

#############################
-Real Data-
######################
Conversation time: {dialogue_time}
Text: {input_text}
######################
Output:
"""


def build_extraction_user_prompt(content: str, timestamp: str) -> str:
    return UNIFIEDMEM_ENTITY_EXTRACTION_PROMPT.format(
        tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
        record_delimiter=DEFAULT_RECORD_DELIMITER,
        completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
        entity_types=",".join(DEFAULT_ENTITY_TYPES),
        input_text=content,
        dialogue_time=timestamp or "unknown",
    )
