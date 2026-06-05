#!/usr/bin/env python3
"""Generate a large synthetic English corpus for MiniGPT training.

Generates ~100K+ characters of diverse children's stories using templates.
Designed to be run on the GPU server where disk space is available.
"""

import random
import json
import os

# Story templates with slots
CHARACTERS = [
    "Lucy", "Emma", "Oliver", "Sophie", "Jack", "Lily", "Max", "Mia",
    "Noah", "Ava", "Liam", "Zoe", "Ethan", "Chloe", "Lucas", "Grace",
    "Henry", "Ruby", "Oscar", "Ivy"
]

ANIMALS = [
    "rabbit", "cat", "dog", "bird", "fox", "deer", "owl", "squirrel",
    "bear", "hedgehog", "butterfly", "turtle", "frog", "duck", "swan",
    "mouse", "ladybug", "robin", "chipmunk", "otter"
]

COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "white",
    "golden", "silver", "brown", "black", "turquoise", "crimson", "violet"
]

PLACES = [
    "forest", "garden", "meadow", "stream", "mountain", "lake", "village",
    "cave", "valley", "hillside", "castle", "bridge", "island", "beach",
    "pond", "field", "grove", "waterfall", "cliff", "ravine"
]

OBJECTS = [
    "flower", "stone", "feather", "shell", "crystal", "seed", "key",
    "map", "book", "lantern", "compass", "bell", "mirror", "crown",
    "wand", "painting", "necklace", "ring", "scroll", "gem"
]

SEASONS = ["spring", "summer", "autumn", "winter"]

WEATHER = [
    "sunny and warm", "cool and breezy", "rainy and gentle",
    "crisp and clear", "misty and magical", "bright and cheerful"
]

TIMES = [
    "morning", "afternoon", "evening", "night", "dawn", "dusk",
    "twilight", "midnight", "sunrise", "sunset"
]

ADJECTIVES = [
    "beautiful", "magical", "wonderful", "tiny", "gentle", "brave",
    "curious", "kind", "clever", "cheerful", "playful", "quiet",
    "strong", "bright", "peaceful", "mysterious", "ancient", "sparkling"
]

VERBS = [
    "discovered", "explored", "found", "created", "built", "climbed",
    "followed", "watched", "listened to", "gathered", "planted", "painted",
    "sang", "danced", "shared", "helped", "protected", "studied",
    "searched", "admired"
]

TEMPLATES = [
    # Template 1: Discovery story
    """One {time}, {char} went for a walk in the {place}. The weather was {weather}. Everything looked {adj} in the {season} light. {char} walked along the path, looking at the {color} flowers and listening to the birds sing.

Suddenly, {char} noticed something {adj} near a {adj} tree. It was a {adj} {animal} with {color} fur and {color} eyes. The {animal} looked at {char} with curiosity. "{char}," the {animal} seemed to say, "follow me."

{char} followed the {animal} deeper into the {place}. They crossed a {adj} bridge over a {adj} stream. The water was crystal clear, and {color} fish swam lazily beneath the surface. On the other side, they found a {adj} clearing filled with {color} {obj}s.

"This is {adj}!" {char} whispered. The {animal} nodded and picked up a {color} {obj} in its mouth. {char} carefully took the {obj} and put it in their pocket. It felt warm and smooth.

They spent the rest of the {time} exploring together. {char} {verb} the {adj} mushrooms growing on the logs, {verb} the {adj} butterflies dancing in the air, and {verb} the {adj} moss covering the rocks.

When it was time to go home, {char} thanked the {animal} for the wonderful adventure. The {animal} waved its tail and disappeared into the {place}. {char} knew they would come back tomorrow.

That {time}, {char} told their family about the {adj} {animal} and the {color} {obj}. Everyone was amazed. {char} placed the {obj} on their windowsill, where it sparkled in the moonlight. It was the beginning of a {adj} friendship.""",

    # Template 2: Seasonal adventure
    """When {season} came to the {place}, everything changed. The trees turned {color} and {color}, the air grew {adj}, and the {animal}s prepared for the months ahead. {char} loved {season} more than any other time of year.

Every {time} during {season}, {char} would {verb} through the {place}, collecting {color} leaves and {adj} acorns. Their friend, a {adj} {animal} named {friend}, always came along. Together, they {verb} the {adj} paths that wound through the {adj} trees.

One {adj} day, {char} and {friend} {verb} a {adj} {obj} hidden under a pile of {color} leaves. It was {adj} and {color}, unlike anything they had seen before. "What is it?" asked {friend}, tilting their head.

{char} examined it carefully. "I think it's a {adj} {obj}," they said. "It might be {adj}." The {obj} began to {verb} softly, casting a {color} glow all around them.

"Let's take it to the {adj} {place}," suggested {friend}. They carefully carried the {obj} through the {adj} {place}, past the {adj} stream and the {adj} meadow, until they reached the {adj} {place}.

There, they showed the {obj} to the other {animal}s. Everyone was amazed. The {adj} owl said it was a {adj} {obj} from long ago. The {adj} deer said it brought good fortune. The {adj} squirrel said it could grant wishes.

{char} and {friend} decided to share the {obj} with everyone. Whenever someone felt sad or lonely, the {obj} would {verb} and fill them with warmth. It became the most {adj} thing in the entire {place}.

As {season} turned to {next_season}, the {obj} grew even more {adj}. It was a reminder that the best adventures are the ones shared with friends.""",

    # Template 3: Friendship story
    """{char} had always been {adj}. While other children played loud games, {char} preferred to {verb} quietly by the {adj} {place}, watching the clouds drift by and listening to the wind whisper through the {color} trees.

One {adj} {time}, {char} {verb} a {adj} {animal} sitting alone by the {adj} {place}. The {animal} looked {adj} and lost. {char} approached slowly and sat down nearby. "Are you okay?" {char} asked gently.

The {animal} looked up with {color} eyes. "I'm {friend}," said the {animal} softly. "I've lost my way home. I live beyond the {adj} {place}, past the {adj} {place} and through the {adj} {place}."

{char} smiled warmly. "I'll help you find your way," they said. "I know these {place}s better than anyone." And so began the {adj} journey of {char} and {friend}.

They {verb} over the {adj} hills, where {color} wildflowers grew in every direction. They {verb} across the {adj} stream, hopping from stone to stone. They {verb} through the {adj} {place}, where ancient trees told stories with their rustling leaves.

Along the way, they met many {adj} creatures. A {color} butterfly showed them the safest path. A {adj} frog told them which mushrooms were safe to eat. A {adj} bird sang them a song about courage and friendship.

When they finally reached {friend}'s home, the sun was setting in a {adj} display of {color} and {color} and {color}. {friend}'s family was overjoyed. They invited {char} to stay for a {adj} feast of berries and honey.

From that day on, {char} and {friend} were the closest of friends. They {verb} together every {time}, exploring new paths and discovering new wonders. And they always remembered the {adj} journey that brought them together.""",

    # Template 4: Magical quest
    """In the {adj} village of {place}, there lived a {adj} child named {char}. {char} was known throughout the village for being {adj} and {adj}. Everyone said that {char} had a {adj} gift for finding lost things.

One {adj} {season} {time}, the village elder called {char} to the {adj} {place}. "Our most {adj} {obj} has gone missing," the elder said with worry. "Without it, the {place} will lose its {adj} magic. Will you help us find it?"

{char} nodded bravely. "I will find the {obj}," they promised. The elder gave {char} a {color} {obj} that would light the way. "This will guide you," the elder said. "Follow its {color} glow."

And so {char} set off into the {adj} {place}. The {color} {obj} {verb} in their hand, casting a {adj} light on the path ahead. The trees grew {adj} and the shadows grew {color}, but {char} was not afraid.

Soon, {char} met a {adj} {animal}. "I know where the {obj} is," said the {animal}. "It was taken by a {adj} {animal} who lives in the {adj} {place}. But be warned — the path is {adj} and full of {adj} surprises."

{char} thanked the {animal} and continued. They {verb} over {adj} rivers, {verb} under {adj} branches, and {verb} through {adj} meadows. The {color} {obj} grew {adj} as they got closer.

Finally, {char} reached the {adj} {place}. There, sitting on a {adj} {obj}, was the {adj} {animal} holding the village's {adj} {obj}. "Why did you take it?" {char} asked.

"I didn't mean to cause trouble," said the {animal} softly. "I just wanted something {adj} to look at. Everything in my {place} is so {adj} and {adj}."

{char} smiled kindly. "You're welcome to visit our village anytime," they said. "The {obj} belongs to everyone." The {animal} was overjoyed and promised to return it.

Together, {char} and the {adj} {animal} brought the {obj} back to the village. Everyone celebrated with a {adj} feast. The {adj} {obj} {verb} once more, filling the {place} with {color} light and {adj} warmth. And {char} had made a new friend.""",

    # Template 5: Nature exploration
    """The {adj} {place} was {char}'s favorite place in the whole world. Every {season}, it transformed into something new and {adj}. In the {time}, the {color} light would filter through the {adj} canopy, creating {adj} patterns on the {adj} ground.

{char} kept a journal of all the {adj} things they {verb} in the {place}. Today's entry was about the {adj} {animal} they had {verb} near the {adj} {place}. It had {color} feathers and {color} eyes, and it sang the most {adj} song.

{char} {verb} the song in their journal, trying to capture its {adj} melody. They also drew a {adj} picture of the {animal}, using {color} and {color} and {color} crayons. It was their best drawing yet.

As {char} walked deeper into the {place}, they {verb} a {adj} {obj} growing between two {adj} rocks. It was {color} and {adj}, with {adj} petals that sparkled like tiny stars. {char} had never seen anything like it.

They carefully sketched the {obj} in their journal and wrote a {adj} description. "Found near the {adj} {place}," they wrote. "Color: {color}. Size: {adj}. Smells like {adj} rain and {adj} sunshine."

Further along the path, {char} {verb} a {adj} {animal} building a {adj} nest in a {color} bush. The {animal} worked {adj} and {adj}, weaving {color} twigs and {adj} leaves together. {char} watched in {adj} silence for a long time.

When the {time} turned to {time}, {char} headed home with a journal full of {adj} observations. They {verb} about the {adj} {obj}, the {adj} {animal}, and the {adj} nest. Tomorrow, they would return with new questions and new wonders to {verb}.

That {time}, as {char} lay in bed, they could hear the {adj} sounds of the {place} through their window. The {animal}s were singing their {adj} songs, the wind was {verb} through the {adj} trees, and somewhere in the {adj} {place}, the {color} {obj} was {verb} softly in the moonlight.""",
]


def generate_corpus(n_stories=200, seed=42):
    """Generate a corpus of synthetic children's stories."""
    random.seed(seed)
    stories = []

    for i in range(n_stories):
        template = random.choice(TEMPLATES)
        char = random.choice(CHARACTERS)
        friend = random.choice([c for c in CHARACTERS if c != char])

        story = template.format(
            char=char,
            friend=friend,
            animal=random.choice(ANIMALS),
            color=random.choice(COLORS),
            place=random.choice(PLACES),
            obj=random.choice(OBJECTS),
            season=random.choice(SEASONS),
            next_season=SEASONS[(SEASONS.index(random.choice(SEASONS)) + 1) % 4],
            weather=random.choice(WEATHER),
            time=random.choice(TIMES),
            adj=random.choice(ADJECTIVES),
            verb=random.choice(VERBS),
        )
        stories.append(story)

    corpus = "\n\n".join(stories)
    return corpus


def main():
    corpus = generate_corpus(n_stories=300)
    output_path = os.environ.get('OUTPUT_PATH', 'generated_corpus.txt')
    with open(output_path, 'w') as f:
        f.write(corpus)
    print(f"Generated corpus: {len(corpus)} chars, {corpus.count(chr(10))} lines")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
