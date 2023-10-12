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

REASONING_QA_FEWSHOT_EXAMPLES = [
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
                "question": "How would a music producer create the sounds heard in this track?",
                "answer": "To create the sounds heard in this track, a music producer would use a combination of instruments and techniques. For the electric guitar part, the producer would need an electric guitar, an amplifier, and potentially some effects pedals to achieve the desired tone. The descending run, arpeggiated chord, double stop hammer on, and sliding notes can be achieved by playing the guitar in a specific manner. The percussion can be created using a drum kit, and in this case, rim shots are utilized to produce a distinctive sound. The microphone should be placed close to the drum where the rim shots are being played in order to fully capture this important sound. The bass part is relatively simple, requiring a bass guitar or a synthesizer capable of producing bass sounds. The piano part can be played on a piano or a digital keyboard with piano sounds. Overall, a producer would need a recording setup, including microphones, audio interface, and digital audio workstation software, to capture and mix all the elements together to create the final track.",
            },
            {
                "question": "What genre does this song belong to?",
                "answer": "Given the instrumentation and the relaxed mood described, this song exhibits elements of a country feel. It has an electric guitar as the main instrument, which is often associated with country music. Additionally, the slow tempo and the overall atmosphere of the track contribute to the country vibe.",
            },
            {
                "question": "Are the piano and guitar playing at the same time?",
                "answer": "Yes, the piano and the guitar are playing at the same time. The piano is providing backing chords to support the guitar's melodic and rhythmic elements. This simultaneous playing creates a harmonic foundation and adds depth to the overall sound of the song.",
            },
            {
                "question": "Could you recommend any songs or artists with a similar style to this track?",
                "answer": "If you enjoy the relaxing and country-infused atmosphere of this track, you might find the following songs or artists appealing: 'Wagon Wheel' by Old Crow Medicine Show, 'Landslide' by Fleetwood Mac, 'Horse with No Name' by America, 'Country Roads' by John Denver, or 'The Weight' by The Band. These songs share a similar acoustic or electric guitar-driven sound, mellow mood, and a touch of country influence. Exploring the works of these artists and their respective discographies might lead you to more music that resonates with the style you're looking for.",
            },
            {
                "question": "How would someone learn to play this type of music?",
                "answer": "To learn to play this type of music, one could take several steps. First, acquiring a basic understanding of guitar playing would be helpful since the guitar is the main instrument in this track. This could involve learning chords, scales, and various guitar techniques through online tutorials, instructional books, or taking lessons from a guitar teacher. Once a foundation is established, one could study specific techniques and patterns commonly found in country or country-influenced music, such as fingerpicking, slide guitar, or pedal steel techniques. Exploring the works of influential guitarists in the country genre, like Chet Atkins or Brad Paisley, can provide valuable inspiration and insight. Additionally, familiarizing oneself with common chord progressions and scales used in country music would be beneficial. Finally, practicing by playing along with similar songs or backing tracks can help develop the necessary skills and improve overall musicianship.",
            },
            {
                "question": "How could someone change this music to make it sound happier?",
                "answer": "To make this music sound happier, several changes could be made. One approach would be to modify the chord progression by incorporating major chords or using more uplifting chord voicings. Changing the tempo to a slightly faster pace can also contribute to a more upbeat atmosphere. Introducing brighter-sounding instruments like acoustic guitars or adding percussive elements such as tambourine or shakers can enhance the happiness of the track. Additionally, adjusting the mix to emphasize higher frequencies and adding melodic elements that convey a sense of joy or playfulness can further enhance the desired emotional effect. Overall, by altering the harmonic, melodic, and timbral elements of the composition, one can transform the mood of the music to be more cheerful and uplifting.",
            },
            {
                "question": "How would you instruct an AI to generate music that is similar to this song?",
                "answer": "To instruct an AI to generate music similar to this song, you could provide the following instructions: 'Produce an instrumental track featuring an electric guitar as the main instrument. Begin with a descending run followed by an arpeggiated chord, a double stop hammer on to a higher note, and a descending slide. Accompany the guitar with a simple percussion beat using rim shots and a bass playing a single note on the first count of each bar. Add a piano playing backing chords to support the guitar. Maintain a relaxing mood throughout the track. Avoid any vocal elements. Aim for a country feel in the overall sound. The tempo should be slow. Consider the track suitable for a coffee shop ambiance.' These instructions would guide the AI in generating music that shares similar characteristics with the provided audio clip.",
            },
        ],
    ),
    # 'ytid': '1KN3GrwhY8c'
    FewShotExample(
        user=[
            {
                "caption": "This disco song features a male voice singing the main melody. This is accompanied by programmed percussion playing a disco beat. The kick is played at every count of the bar. The synth changes chords after two bars to a higher pitch chord, then after one bar to a higher pitch chord, then after one more bar to a higher pitch chord and finally after one more bar to a suspended fourth chord. The instruments pause for 3 counts of the bar while the voice continues to sing. The bass plays the root notes of the chords. The mood of this song is romantic. This song can be played in a club.",
                "aspect_list": "['disco song', 'male voice', 'synth sounds', 'programmed percussion', 'bass', 'moderate tempo', 'upbeat', 'romantic theme', 'dance song']",
            }
        ],
        assistant=[
            {
                "question": "In the disco beat, what are the other percussion instruments besides the kick drum?",
                "answer": "In addition to the kick drum, the disco beat may include other percussion instruments such as hi-hats, snare drums, cymbals, and tambourines. These instruments contribute to the rhythmic texture and add variety to the groove.",
            },
            {
                "question": "During the instrumental break, does the male voice continue singing the main melody or is it completely silent?",
                "answer": "During the instrumental break, the male voice continues to sing the main melody. While the instruments pause for three counts of the bar, the vocal performance maintains continuity and provides a melodic focal point for the listeners.",
            },
            {
                "question": "What elements could be changed in this song to make it sound more danceable?",
                "answer": "To make this song sound more danceable, certain elements could be modified. The tempo could be slightly increased to create a more energetic pace. The programmed percussion could be enhanced with additional layers and variations to make the beat more intricate and compelling. Adding syncopated rhythmic patterns or accentuating certain beats could also enhance the danceability. Additionally, incorporating melodic hooks or catchy vocal phrases can further engage the audience on the dancefloor.",
            },
            {
                "question": "What are some notable songs that share a theme and genre similar to this song?",
                "answer": "Some notable disco songs that share a romantic theme similar to this song include 'Love's Theme' by Barry White's Love Unlimited Orchestra, 'I Love the Nightlife' by Alicia Bridges, 'You Should Be Dancing' by Bee Gees, 'Funky Town' by Lipps Inc., and 'Never Can Say Goodbye' by Gloria Gaynor. These songs, like the one described, combine disco's infectious beats with lyrics that express themes of love and romance.",
            },
            {
                "question": "How could a music producer modify this song to give it a happier vibe without changing the fundamental structure?",
                "answer": "To give this song a happier vibe without altering its fundamental structure, a music producer could focus on several aspects. They could introduce brighter and more uplifting synth sounds or use major chord progressions instead of minor ones to create a more joyful tonality. Adjusting the mix to emphasize higher frequencies and adding shimmering effects could enhance the overall brightness. Additionally, incorporating energetic and lively vocal melodies or harmonies can contribute to a happier atmosphere. Careful attention to the overall arrangement and production choices can shape the emotional impact and evoke a sense of positivity and joy.",
            },
        ],
    ),
]
