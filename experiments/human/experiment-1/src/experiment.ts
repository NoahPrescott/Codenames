import jsPsychHtmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response'
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response'
import jsPsychSurveyText from '@jspsych/plugin-survey-text'
import jsPsychSurveyMultiChoice from '@jspsych/plugin-survey-multi-choice'
import jsPsychHtmlSliderResponse from '@jspsych/plugin-html-slider-response'

//import { pair_sets } from './word_pairs.js'
//import custom plug-in
//import HtmlChoicePlugin from './plugin-html-choice
//import HTMLMultiButtonResponse from '../jspsych-contrib/packages/plugin-html-multi-button-response/dist/index.js'

//import HtmlChoicePlugin from "@jspsych-contrib/plugin-html-choice";

//import jsPsychPreload from '@jspsych/plugin-preload'
import { initJsPsych } from 'jspsych'

import { saveTrialDataComplete, saveTrialDataPartial } from './databaseUtils'
import { debugging, getExptInitialized, getUserInfo, prolificCC, prolificCUrl } from './globalVariables'

import type { KeyboardResponse, Task, TrialData } from './project'
import type { DataCollection } from '../node_modules/jspsych/dist/modules/data/DataCollection'

import { prepareExperimentWordPairs } from './databaseUtils';
const debug = debugging()

const debuggingText = debug ? `<br /><br />redirect link : ${prolificCUrl}` : '<br />'
const exitMessage = `<p class="align-middle text-center"> 
Please wait. You will be redirected back to Prolific in a few moments. 
<br /><br />
If not, please use the following completion code to ensure compensation for this study: ${prolificCC}
${debuggingText}
</p>`

const exitExperiment = () => {
  document.body.innerHTML = exitMessage
  setTimeout(() => {
    window.location.replace(prolificCUrl)
  }, 3000)
}
const exitExperimentDebugging = () => {
  const contentDiv = document.getElementById('jspsych-content')
  if (contentDiv) contentDiv.innerHTML = exitMessage
}

let word_pairs: any[] = [];
//console.log(word_pairs)

export async function runExperiment() {

  if (word_pairs.length === 0) {
    word_pairs = await prepareExperimentWordPairs();
  }
  if (debug) {
    console.log('--runExperiment--')
    console.log('UserInfo ::', getUserInfo())
  }

  /* initialize jsPsych */
  const jsPsych = initJsPsych({
    show_progress_bar: true,
    on_data_update: function (trialData: TrialData) {
      console.log("data updating")
      trialData.saveToFirestore = true;
      if (debug) {
        console.log("x" + trialData.saveToFirestore)
        console.log('jsPsych-update :: trialData ::', trialData)
      }
      // if trialData contains a saveToFirestore property, and the property is true, then save the trialData to Firestore
      if (trialData.saveToFirestore) {
        console.log("saving to firestore", trialData.saveToFirestore)
        saveTrialDataPartial(trialData).then(
          () => {
            if (debug) {
              console.log('saveTrialDataPartial: Success') // Success!
            }
          },
          (err) => {
            console.error(err) // Error!
          },
        )
      }
    },
    on_finish: (data: DataCollection) => {
      const contentDiv = document.getElementById('jspsych-content')
      if (contentDiv) contentDiv.innerHTML = '<p> Please wait, your data are being saved.</p>'
      saveTrialDataComplete(data.values()).then(
        () => {
          if (debug) {
            exitExperimentDebugging()
            console.log('saveTrialDataComplete: Success') // Success!
            console.log('jsPsych-finish :: data ::')
            console.log(data)
            setTimeout(() => {
              jsPsych.data.displayData()
            }, 3000)
          } else {
            exitExperiment()
          }
        },
        (err) => {
          console.error(err) // Error!
          exitExperiment()
        },
      )
    },
  })
  //experiment vars
  var completion_code = 'xxx';
  var trial_order = 0;
  //var nTrials = pair_sets.length;

  var nTrials = 80;
  console.log("number of trials: ", nTrials)

  // consent
  var consent = {
      type: jsPsychHtmlButtonResponse,
      stimulus: "<DIV align='left'><div>&nbsp;</div><div>Please consider this information carefully before deciding whether to participate in this research.</div><div>&nbsp;</div><div>The purpose of this research is to examine which factors influence social judgment and decision-</div><div>making. You will be asked to make judgements about individuals and actions in social scenarios.</div><div>We are simply interested in your judgement. The study will take less than 1 hour to complete,</div><div>and you will receive less than $20 on Amazon Mechanical Turk. Your compensation and time</div><div>commitment are specified in the study description. There are no anticipated risks associated with</div><div>participating in this study. The effects of participating should be comparable to those you would</div><div>ordinarily experience from viewing a computer monitor and using a mouse or keyboard for a</div><div>similar amount of time. At the end of the study, we will provide an explanation of the questions</div><div>that motivate this line of research and will describe the potential implications.</div><div>&nbsp;</div><div>Your participation in this study is completely voluntary and you may refuse to participate or you</div><div>may choose to withdraw at any time without penalty or loss of benefits to you which are</div><div>otherwise entitled. Your participation in this study will remain confidential. No personally</div><div>identifiable information will be associated with your data. Also, all analyses of the data will be</div><div>averaged across all the participants, so your individual responses will never be specifically</div><div>analyzed.</div><div>&nbsp;</div><div>If you have questions or concerns about your participation or payment, or want to request a</div><div>summary of research findings, please contact Dr. Jonathan Phillips at</div><div><a href=mailto:Jonathan.S.Phillips@dartmouth.edu>Jonathan.S.Phillips@dartmouth.edu</a>.</div><div>&nbsp;</div><div>Please save a copy of this form for your records.</div><div>&nbsp;</div></DIV><div>Agreement:</div><DIV align='left'><div>The nature and purpose of this research have been sufficiently explained and I agree to</div><div>participate in this study. I understand that I am free to withdraw at any time without incurring</div><div>any penalty. Please consent by clicking the button below to continue. Otherwise, please exit the</div><div>study at any time.</div><div>&nbsp;</div></DIV>",
      choices: ['Submit']
  };

  var welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: "<div class='center-content'><br><br><br><br>Welcome to the HIT. Press any key to begin.",
  };

  //get subject ID
  var get_id = {
    type: jsPsychSurveyText,
      preamble: ["Please enter your Prolific Worker ID below.<br><br>If you do not enter your ID accurately, we will not be able to pay you."],
      questions: [{prompt: "Worker ID:", name: "subject_id", required: true}],
      on_finish: function(data: TrialData){
          jsPsych.data.addProperties({
            completion_code: completion_code,
            subject_id: data['response']['subject_id'],
          });
      }
  }
  
  //set instructions
  var instructions = {
      type: jsPsychHtmlButtonResponse,
      stimulus: "In this game, you will be shown two words and asked to rate how related they are. <br><br> There are " + nTrials + " rounds, with one pair of words per round. <br><br> Click 'Continue' to begin!<br><br>",
      choices: ['Continue']
  };
  //subject chooses words and gives clue
  var trial = {
      type: jsPsychHtmlSliderResponse,
      stimulus: function() {
        var pairIndex = trial_order;
        var word1 = word_pairs[pairIndex].word1;
        var word2 = word_pairs[pairIndex].word2;
        return `<b>${word1}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${word2}</b>`;
      },
      prompt: "Please indicate how related these two words are.<b><br><br>", 
      labels: ["not related","very related"],
      require_movement: true,
      slider_start: 0,
      on_finish: function (data: TrialData) {
          data.exp_phase = 'trial';
          data.trial_order = trial_order+1;
          data.word1 = word_pairs[trial_order].word1;
          data.word2 = word_pairs[trial_order].word2;
          data.saveToFirestore = true;
          trial_order=trial_order+1;
      }
  };
  
  //demographics  
  var demo_instructions = {
      type: jsPsychHtmlKeyboardResponse,
    stimulus: "<div> Thank you!</div><div>Now, please provide us with some demographic information.</div><div>Press any key to begin.</div>",
  };
  var demo1 = {
      type: jsPsychSurveyText,
      preamble: '',
      questions: [{prompt: "How old are you?", required: true}, {prompt: "What is your native language?", required: true}, {prompt: "What is your nationality?", required: true}, {prompt: "In which country do you live?", required: true}],
  };
  //saves data on completion of this trial
  var demo2 = {
      type: jsPsychSurveyMultiChoice,
      preamble: "Please provide us with some demographic information.",
      questions: [
          {prompt: "What is your gender?", options: ["Male","Female","Other"], required:true}, 
          {prompt: "Are you a student?", options: ["Yes","No"], required: true},
          {prompt: "What is your education level?", options: ["Grade/elementary school","High school","Some college or university","College or university degree","Graduate degree, Masters","PhD"], required: true}
      ],
      on_finish: function(data: TrialData) {
          data.exp_phase = 'subject_info';
          var lastData = jsPsych.data.get().last(2).values();
          var demo1 = lastData[0]['response'];
          var demo2 = lastData[1]['response'];
          data.age = demo1['Q0'];
          data.language = demo1['Q1'];
          data.nationality = demo1['Q2'];
          data.country = demo1['Q3'];
          data.gender = demo2['Q0'];
          data.student = demo2['Q1'];
          data.education = demo2['Q2'];
          save_data(data);
      },
};

  //provide completion code


  //debrief
  var debrief = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: "<DIV align='left'><div>&nbsp;</div><div><strong>Study Debriefing</strong></div><div>Judgement and decision making are important aspects of public and private life. Using surveys</div><div>like the one you just completed, we are examining the factors that go into making social</div><div>decisions.</div><div>&nbsp;</div><div><strong>How is this being tested?</strong></div><div>We have asked you to respond to stories or videos that differ on several important factors. By</div><div>isolating different variables that are involved in social thought, we can better understand how we</div><div>arrive at complex decision-making. For example, some actions are seen as more worthy of blame</div><div>if they are performed intentionally. Harming someone on purpose is generally worse than</div><div>harming someone by accident, or even harming someone in a way that is foreseen but not</div><div>intended.</div><div>&nbsp;</div><div><strong>Main questions and hypotheses:</strong></div><div>A fundamental goal of our research is to understand the cognitive and emotional factors that</div><div>influence social judgment and decision-making. We are studying these factors by presenting</div><div>people with hypothetical questions that vary in specific ways and seeing which factors make a</div><div>difference. Some people filled out the same survey that you just filled out. Others got slightly</div><div>different surveys.</div><div>&nbsp;</div><div><strong>Why is this important to study?</strong></div><div>By comparing answers on these important factors, we learn about what factors affect social</div><div>judgment. This has crucial implications for many public domains, including the legal system.</div><div>&nbsp;</div><div><strong>How to learn more:</strong></div><div>If you are interested in learning more, you may want to consult the following articles:</div><div>Phillips, J., &amp; Cushman, F. (2017). Morality constrains the default representation of what is</div><div style='padding-left: 30px;'>possible. Proceedings of the National Academy of Sciences of the United States of</div><div style='padding-left: 30px;'>America, 114(18), 4649&ndash;4654. https://doi.org/10.1073/pnas.1619717114</div><div>Phillips, J., Morris, A., &amp; Cushman, F. (2019). How we know what not to think.</div><div style='padding-left: 30px;'>Trends in Cognitive Sciences, 23(12), 1026&ndash;1040. https://doi.org/10.1016/j.tics.2019.09.007</div><div>Phillips, J., Buckwalter, W., Cushman, F., Friedman, O., Martin, A., Turri, J., Santos, L., &amp;</div><div style='padding-left: 30px;'>Knobe, J. (2020). Knowledge before Belief. Behavioral and Brain Sciences, 1-37.</div><div style='padding-left: 30px;'>doi:10.1017/S0140525X20000618</div><div>&nbsp;</div><div><strong>How to contact the researcher:</strong></div><div>If you have questions or concerns about your participation or</div><div>payment, or want to request a summary of research findings, please contact the Primary</div><div>Investigator: Dr. Jonathan Phillips, at Jonathan.S.Phillips@dartmouth.edu.</div><div>Whom to contact about your rights in this research:</div><div>If you have questions, concerns,</div><div>complaints, or suggestions about the present research, you may call the Office of the Committee</div><div>for the Protection of Human Subjects at Dartmouth College (603) 646-6482 during normal</div><div>business hours.</div><div>&nbsp;</div><div><strong>Thank you for your participation!</strong></div><div>&nbsp;</div></DIV>",
  };

  //save data to database
  function save_data(data: TrialData) {
      var url = "https://judgement-comp.web.app/"; //use this when running on server
      //var url = "/"; //use this when running locally
      var xhr = new XMLHttpRequest();
      xhr.open("POST", url, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.send(JSON.stringify({
          data
      }));
  }

  //set timeline
  var timeline = [];

  timeline.push(consent);
  timeline.push(welcome);
  timeline.push(get_id);
  timeline.push(instructions);
  for (var i = 0; i < nTrials; i++) {
      timeline.push(trial);
  };
  timeline.push(demo_instructions)
  timeline.push(demo1);
  timeline.push(demo2);
  //timeline.push(debrief);

  // jsPsych.init({
  //     timeline:timeline,
  // });

  /* start the experiment */
  await jsPsych.run(timeline)
}
