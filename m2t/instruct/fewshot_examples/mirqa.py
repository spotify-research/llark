# Copyright 2023 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from m2t.instruct.fewshot_examples import FewShotExample

MIRQA_FEWSHOT_EXAMPLES = [
    # yt id: -0SdAVK79lg
    FewShotExample(
        user=[
            {
                "aspect_list": "['guitar song', 'piano backing', 'simple percussion', 'relaxing melody', 'slow tempo', 'bass', 'country feel', 'instrumental', 'no voice']",
                "caption": "This song features an electric guitar as the main instrument. The guitar plays a descending run in the beginning then plays an arpeggiated chord followed by a double stop hammer on to a higher note and a descending slide followed by a descending chord run. The percussion plays a simple beat using rim shots. The percussion plays in common time. The bass plays only one note on the first count of each bar. The piano plays backing chords. There are no voices in this song. The mood of this song is relaxing. This song can be played in a coffee shop.",
            }
        ],
        assistant=[
            {
                "question": "What is the main instrument in this song?",
                "answer": "The main instrument in this song is an electric guitar.",
            },
            {
                "question": "What does the guitar play in the beginning?",
                "answer": "In the beginning, the guitar plays a descending run. This means that the guitar plays a sequence of notes decreasing in pitch.",
            },
            {
                "question": "Can you describe the guitar's playing after the descending run?",
                "answer": "After the descending run, the guitar plays an arpeggiated chord followed by a double stop hammer on to a higher note and a descending slide followed by a descending chord run.",
            },
            {
                "question": "What is the percussion instrument used in this song?",
                "answer": "The percussion instrument used in this song is a drum set. The drummer plays a simple beat using rim shots.",
            },
            {
                "question": "What time signature does the percussion play in?",
                "answer": "The percussion plays in common time, which is also known as 4/4 time.",
            },
            {
                "question": "What does the bass play?",
                "answer": "The bass plays only one note on the first count of each bar.",
            },
            {
                "question": "What role does the piano play in this song?",
                "answer": "The piano plays backing chords in this song. This means that it is playing the role of accompanying the lead instrument, which in this case is the guitar, by establishing the harmonic structure of the song.",
            },
            {
                "question": "Are there any vocals in this song?",
                "answer": "No, there are no vocals in this song.",
            },
            {
                "question": "How would you describe the overall mood and emotions conveyed by the song?",
                "answer": "The overall mood of this song is relaxing. It features a slow tempo, a country feel, and a melody that evokes a sense of calmness and tranquility. The absence of vocals allows the listener to focus on the instrumental elements and creates a soothing atmosphere. This song would be fitting for a coffee shop environment, enhancing a laid-back and peaceful ambiance.",
            },
            {
                "question": "Is there a cello in this song?",
                "answer": "No, there is not an audible cello in this song.",
            },
            {
                "question": "Is there a synthesizer in this song?",
                "answer": "No, there does not appear to be a synthesizer in this song.",
            },
            {
                "question": "What genre would you classify this song as?",
                "answer": "This song can be classified as an instrumental piece with elements of country music. The presence of guitar, piano, bass, and simple percussion contribute to the country feel.",
            },
            {
                "question": "What is the approximate tempo of the clip in beats per minute (BPM)?",
                "answer": "The tempo of this song is slow. However, it is difficult to determine the precise tempo of the song. A typical BPM range for a slow country ballad would be between 70-90 BPM.",
            },
        ],
    ),
    # ytid vy2gUzgmBzQ
    FewShotExample(
        user=[
            {
                "caption": "This goth metal song features a male voice singing the main melody. The voice is deep. Toward the end, there are other male voices singing vocables. This is accompanied by heavy percussion using a five-beat common time pattern played on floor toms with a shuffle note each note creating a triplet feel. The distortion guitars play heavy descending chord riffs using palm muting technique. The bass guitar plays the root notes of the chords. At the background, there is an high pitched sound played on a synth that gives this song an eerie feel. This song can be played at the end credits of a horror movie.",
                "aspect_list": "['heavy metal', 'goth metal', 'male voice', 'amateur recording', 'distortion guitars', 'heavy percussion', 'descending chords', 'synth backing', 'eerie']",
            }
        ],
        assistant=[
            {
                "question": "What is the main genre of this song?",
                "answer": "The main genre of this song is goth metal.",
            },
            {
                "question": "What type of voice is singing the main melody?",
                "answer": "A deep male voice is singing the main melody.",
            },
            {
                "question": "Are there any additional vocals in the clip?",
                "answer": "Yes, toward the end of the clip, there are other male voices singing vocables.",
            },
            {
                "question": "What instruments are used for the heavy percussion in the song?",
                "answer": "The heavy percussion in the song is created using floor toms.",
            },
            {
                "question": "How many beats are in each measure of the common time pattern used in the percussion?",
                "answer": "The common time pattern used in the percussion has five beats in each measure.",
            },
            {
                "question": "What technique is used by the distortion guitars?",
                "answer": "The distortion guitars play heavy descending chord riffs using palm muting technique.",
            },
            {
                "question": "What role does the bass guitar play in this song?",
                "answer": "The bass guitar plays the root notes of the chords.",
            },
            {
                "question": "Is there any synth used in this song?",
                "answer": "Yes, there is a high-pitched sound played on a synth in the background.",
            },
            {
                "question": "Is there any violin used in this song?",
                "answer": "No, there is no violin audible in this clip.",
            },
            {
                "question": "What atmosphere does the synth contribute to the song?",
                "answer": "The synth contributes to the song's eerie feel.",
            },
            {
                "question": "What is the suggested usage of this song?",
                "answer": "This song can be played at the end credits of a horror movie.",
            },
            {
                "question": "List all instruments audible in the track.",
                "answer": "There are vocals, heavy percussion (drum set), distorted guitar, bass guitar, and synthesizer in this song.",
            },
        ],
    ),
    # ytid ZleHXDirD58
    FewShotExample(
        user=[
            {
                "caption": "This symphonic song features an orchestra playing a variety of instruments. The main melody is played on trumpets and violins. This is accompanied by the brass section and horn section of the orchestra. Clarinets play in harmony with the violins and trumpets. A xylophone kind of instrument plays bell sounds with the same melody as the main instruments. Cellos play the bass part of the song. This is accompanied by a bass drum playing a steady count. A drum kit plays the percussion part with a marching beat. There are no voices in this song. This is an instrumental song. The feel of this song is uplifting.",
                "aspect_list": "['orchestra music', 'symphonic song', 'amateur recording', 'percussion', 'violins', 'trumpets', 'cello', 'flute', 'clarinets', 'moderate tempo', 'instrumental', 'no voices', 'uplifting music']",
            }
        ],
        assistant=[
            [
                {
                    "question": "What are the main instruments playing the melody in the song?",
                    "answer": "The main melody in the song is played on trumpets and violins.",
                },
                {
                    "question": "What other sections of the orchestra accompany the main melody?",
                    "answer": "The brass section and horn section of the orchestra accompany the main melody.",
                },
                {
                    "question": "List all of the instruments present in the song.",
                    "answer": "This song includes a brass section, a horn section, clarinets, violins, trumpets, xylophone, cello, flute, bass drum, and drum kit.",
                },
                {
                    "question": "Are there any instruments playing in harmony with the violins and trumpets?",
                    "answer": "Yes, clarinets play in harmony with the violins and trumpets.",
                },
                {
                    "question": "Is there any flute being played in the recording?",
                    "answer": "Yes, there is flute present in the song.",
                },
                {
                    "question": "Is there any guitar present in the recording?",
                    "answer": "No, there is no guitar in the recording.",
                },
                {
                    "question": "Is there any instrument playing a bell-like sound in the song?",
                    "answer": "Yes, a xylophone-like instrument plays bell sounds with the same melody as the main instruments.",
                },
                {
                    "question": "What instrument plays the bass part in the song?",
                    "answer": "Cellos play the bass part of the song.",
                },
                {
                    "question": "Is there any percussion in the song?",
                    "answer": "Yes, there is percussion in the song. A drum kit plays the percussion part with a marching beat, and a bass drum plays a steady count.",
                },
                {
                    "question": "Are there any vocals in the song?",
                    "answer": "No, there are no vocals in this song. It is an instrumental song.",
                },
                {
                    "question": "How would you describe the overall feel of the song?",
                    "answer": "The song has an uplifting feel.",
                },
                {
                    "question": "What is the tempo of the song?",
                    "answer": "The tempo of the song is moderate.",
                },
                {
                    "question": "What style or genre does this song belong to?",
                    "answer": "This song probably belongs to the classical, symphonic or orchestral music genre.",
                },
            ]
        ],
    ),
    # ytid V1A4wBgvPgI
    FewShotExample(
        user=[
            {
                "caption": "This jazz song features the main melody played on a flugelhorn and a saxophone. Both instruments play in harmony with variations in melody. These instruments compliment each other. This is accompanied by percussion playing a jazz beat with constant strokes on the ride cymbal. The bass guitar plays a groovy bassline starting with a descending lick. This is then followed with an ascending lick to reach the higher notes before descending back. The guitar plays chords with string muting in a jazz rhythm.  The mood of this song is upbeat. This song can be played in an advertisement for a luxury car.",
                "aspect_list": "['jazz', 'flugelhorn', 'saxophone', 'percussion', 'groovy bass', 'jazz guitar', 'moderate tempo', 'happy mood', 'no voices', 'instrumental']",
            }
        ],
        assistant=[
            {
                "question": "What are the main instruments featured in this jazz song?",
                "answer": "The main instruments featured in this jazz song are the flugelhorn and the saxophone.",
            },
            {
                "question": "How do the flugelhorn and saxophone interact in the song?",
                "answer": "The flugelhorn and saxophone play the main melody in harmony with variations. They complement each other and create a harmonious blend of melodies.",
            },
            {
                "question": "What is the role of percussion in this song?",
                "answer": "The percussion provides a jazz beat and maintains a constant rhythm by playing strokes on the ride cymbal. It adds a rhythmic foundation to the music.",
            },
            {
                "question": "Describe the bassline in this song.",
                "answer": "The bass guitar plays a groovy bassline in this song. It starts with a descending lick and then transitions into an ascending lick to reach higher notes before descending back. This creates a dynamic and melodic bass part.",
            },
            {
                "question": "What is the rhythmic style of the guitar playing in this song?",
                "answer": "The guitar plays chords with string muting in a jazz rhythm. The string muting technique adds a percussive quality to the guitar part, blending well with the overall jazz sound.",
            },
            {
                "question": "How would you describe the mood of this song?",
                "answer": "The mood of this song is upbeat and happy. It has a lively and positive energy that would work well in an advertisement for a luxury car.",
            },
            {
                "question": "Does this song have vocals?",
                "answer": "No, this song is instrumental and does not include any vocals.",
            },
            {
                "question": "What is the approximate tempo of this song in beats per minute (BPM)?",
                "answer": "The tempo of the song could be described as moderate, indicating a moderate pace and groove. The tempo in BPM is around 80-100 BPM.",
            },
            {
                "question": "Are there any vocals in the clip?",
                "answer": "No, there are no vocals in the clip.",
            },
            {
                "question": "Give a complete list of all audible instruments present in the song.",
                "answer": "This song includes a a flugelhorn, saxophone, bass guitar, and jazz guitar.",
            },
            {
                "question": "What is the guitar doing in the song?",
                "answer": "The guitar plays chords with string muting in a jazz rhythm.",
            },
            {
                "question": "What genre does this song belong to?",
                "answer": "This song belongs to the jazz genre, characterized by its improvisational nature, swing rhythms, and harmonic complexity.",
            },
            {
                "question": "How would you describe the overall style of this song?",
                "answer": "The overall style of this song is jazz, which is evident through the instrumentation, harmonic structure, and rhythmic elements present in the music.",
            },
            {
                "question": "What order do the musical events occur in the clip?",
                "answer": "The main melody played on the flugelhorn and saxophone, accompanied by percussion and bass guitar, is established first. The guitar then enters to provide chordal support and rhythmic texture.",
            },
        ],
    ),
]
