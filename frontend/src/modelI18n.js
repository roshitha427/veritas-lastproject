export const modelTranslations = {
  en: {
    visualWarning: 'Experimental visual model: test accuracy is {accuracy}%, below your {target}% target. Its scores need careful review.',
    textDomainWarning: 'Article patterns are not claim verification: this model scored {accuracy}% on {count} short political claims. Its historical article score does not establish accuracy for your claim.',
    aiTextNote: 'This experimental score comes from a model trained on historical human and ChatGPT answers. It is not proof of authorship, and newer generators, paraphrases and other languages may behave differently.',
    aiTextDomainWarning: 'This model scored {test}% on held-out Reddit answers and {domain}% on the separate Wikipedia domain. It is an experimental writing-style estimate, not proof of authorship or a validated result for current news.',
    temporal: 'Learned video consistency model',
    ai_writing: 'Local AI-writing model',ai_generation: 'AI image/frame detector',aiShortText:'AI-writing assessment requires at least {words} English words. Shorter text is left unassessed.',
    scopeTemporal: 'This learned model assesses face-frame consistency using an older face-swap dataset. The main video score differs from the frame chart. It does not establish news truth, whole-video AI authorship or audio authenticity.',
  },
  hi: {
    visualWarning: 'दृश्य मॉडल प्रयोगात्मक है: टेस्ट सटीकता {accuracy}% है, जो आपके {target}% लक्ष्य से कम है। स्कोर की सावधानी से समीक्षा करें।',
    textDomainWarning: 'लेख के पैटर्न दावे का सत्यापन नहीं हैं: इस मॉडल ने {count} छोटे राजनीतिक दावों पर {accuracy}% सटीकता पाई। पुराने लेखों का स्कोर आपके दावे की सटीकता सिद्ध नहीं करता।',
    aiTextNote: 'यह प्रयोगात्मक स्कोर पुराने मानव और ChatGPT उत्तरों पर प्रशिक्षित मॉडल से आता है। यह लेखन का प्रमाण नहीं है। नए जनरेटर, बदले हुए शब्द और अन्य भाषाएँ अलग परिणाम दे सकती हैं।',
    aiTextDomainWarning: 'इस मॉडल ने अलग रखे Reddit उत्तरों पर {test}% और अलग Wikipedia विषय-क्षेत्र पर {domain}% सटीकता पाई। यह प्रयोगात्मक लेखन-शैली अनुमान है, लेखन का प्रमाण या वर्तमान खबरों के लिए सत्यापित परिणाम नहीं।',
    temporal: 'प्रशिक्षित वीडियो संगति मॉडल',
    ai_writing: 'स्थानीय AI लेखन मॉडल',ai_generation: 'AI चित्र/फ़्रेम पहचान',aiShortText:'AI लेखन आकलन के लिए कम से कम {words} अंग्रेज़ी शब्द चाहिए। छोटे टेक्स्ट का आकलन नहीं हुआ है।',
    scopeTemporal: 'यह प्रशिक्षित मॉडल पुराने फेस-स्वैप डेटासेट के आधार पर फ़्रेमों की संगति जाँचता है। मुख्य वीडियो स्कोर फ़्रेम चार्ट से अलग है। यह खबर का सच, पूरे वीडियो के AI सृजन या आवाज़ की प्रामाणिकता तय नहीं करता।',
  },
  te: {
    visualWarning: 'దృశ్య మోడల్ ప్రయోగాత్మకం: టెస్ట్ ఖచ్చితత్వం {accuracy}%, మీ {target}% లక్ష్యం కంటే తక్కువ. స్కోర్లను జాగ్రత్తగా సమీక్షించండి.',
    textDomainWarning: 'వ్యాస నమూనాలు వాదన ధృవీకరణ కాదు: ఈ మోడల్ {count} చిన్న రాజకీయ వాదనలపై {accuracy}% ఖచ్చితత్వం సాధించింది. పాత వ్యాసాల స్కోర్ మీ వాదనకు ఖచ్చితత్వాన్ని నిర్ధారించదు.',
    aiTextNote: 'ఈ ప్రయోగాత్మక స్కోర్ పాత మానవ, ChatGPT సమాధానాలపై శిక్షణ పొందిన మోడల్ నుండి వచ్చింది. ఇది రచయితను నిర్ధారించే రుజువు కాదు. కొత్త జనరేటర్లు, మార్చిన వాక్యాలు, ఇతర భాషల్లో ఫలితాలు మారవచ్చు.',
    aiTextDomainWarning: 'ఈ మోడల్ విడిగా ఉంచిన Reddit సమాధానాలపై {test}%, వేరే Wikipedia రంగంలో {domain}% ఖచ్చితత్వం సాధించింది. ఇది ప్రయోగాత్మక రచనా-శైలి అంచనా మాత్రమే; రచయితకు రుజువు లేదా ప్రస్తుత వార్తలకు ధృవీకరించిన ఫలితం కాదు.',
    temporal: 'శిక్షణ పొందిన వీడియో స్థిరత్వ మోడల్',
    ai_writing: 'స్థానిక AI రచన మోడల్',ai_generation: 'AI చిత్రం/ఫ్రేమ్ గుర్తింపు',aiShortText:'AI రచన అంచనాకు కనీసం {words} ఆంగ్ల పదాలు అవసరం. చిన్న టెక్స్ట్‌ను అంచనా వేయలేదు.',
    scopeTemporal: 'ఈ శిక్షణ పొందిన మోడల్ పాత ఫేస్-స్వాప్ డేటాసెట్ ఆధారంగా ఫ్రేముల స్థిరత్వాన్ని పరీక్షిస్తుంది. ప్రధాన వీడియో స్కోర్ ఫ్రేమ్ చార్ట్‌కు వేరు. ఇది వార్త నిజం, మొత్తం వీడియో AI సృష్టి లేదా ఆడియో ప్రామాణికతను నిర్ధారించదు.',
  },
};
