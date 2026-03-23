import { AnimatePresence, motion } from 'framer-motion';
import { ArrowLeft, BrainCircuit, Clock3, Loader2, Play, Target } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  formatApiError,
  getCognitiveCategoryQuestions,
  getCognitiveReasoningQuestions,
  getCognitiveSpatialRotationQuestions,
  getCognitiveWordBank,
  submitCognitiveTestResults,
} from '../api';


const TOTAL_MODULES = 14;
const MAX_REACTION_ATTEMPTS = 8;
const TEST_DURATION_SECONDS = 10 * 60;

const MODULES = [
  { id: 'memoryEncoding', title: 'Module 1 - Remember Words', copy: 'Remember these words. You will be asked to recall them later.' },
  { id: 'orientation', title: 'Module 2 - Date & Time Questions', copy: 'Answer simple questions about today\'s date.' },
  { id: 'digitForward', title: 'Module 3 - Repeat Numbers', copy: 'Type the numbers in the same order.' },
  { id: 'digitBackward', title: 'Module 4 - Reverse Numbers', copy: 'Type the numbers in reverse order.' },
  { id: 'executiveReasoning', title: 'Module 5 - Pattern Reasoning', copy: 'Pick the best answer to continue the pattern.' },
  { id: 'reactionTime', title: 'Module 6 - Reaction Speed Game', copy: 'Tap each target as quickly as you can.' },
  { id: 'visualPattern', title: 'Module 7 - Pattern Memory Grid', copy: 'Watch the pattern, then rebuild it from memory.' },
  { id: 'verbalFluency', title: 'Module 8 - Animal Naming Task', copy: 'Name as many animals as you can in 30 seconds.' },
  { id: 'categoryMatching', title: 'Module 9 - Odd Item Out', copy: 'Choose the item that does not belong.' },
  { id: 'stroop', title: 'Module 10 - Color Word Test', copy: 'Choose the text color, not the word itself.' },
  { id: 'symbolDigit', title: 'Module 11 - Symbol Matching', copy: 'Use the key to match each symbol to a number.' },
  { id: 'mentalArithmetic', title: 'Module 12 - Quick Math', copy: 'Solve these quick subtraction questions.' },
  { id: 'spatialRotation', title: 'Module 13 - Shape Rotation', copy: 'Choose the shape that matches the clockwise or anticlockwise rotation.' },
  { id: 'delayedRecall', title: 'Module 14 - Word Recall', copy: 'Type the words you remember from Module 1.' },
];

const MONTH_OPTIONS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

const WEEKDAY_OPTIONS = [
  'Sunday',
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
];

const STROOP_COLOR_CLASSES = {
  red: 'text-rose-400',
  blue: 'text-sky-400',
  green: 'text-emerald-400',
  yellow: 'text-amber-300',
};


function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}


function createSeededRandom(seed) {
  let state = seed % 2147483647;
  if (state <= 0) {
    state += 2147483646;
  }

  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}


function shuffleWithRandom(items, random) {
  const list = [...items];
  for (let index = list.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [list[index], list[swapIndex]] = [list[swapIndex], list[index]];
  }
  return list;
}


function pickRandom(items, count, random) {
  if (!Array.isArray(items) || items.length === 0) {
    return [];
  }
  return shuffleWithRandom(items, random).slice(0, count);
}


function buildDigitSequence(length, random) {
  const sequence = [];
  for (let index = 0; index < length; index += 1) {
    sequence.push(String(Math.floor(random() * 10)));
  }
  return sequence;
}


function normalizeWords(inputText) {
  if (!inputText || !inputText.trim()) {
    return [];
  }

  const unique = new Set(
    inputText
      .toLowerCase()
      .split(/[\s,;\n]+/)
      .map((word) => word.trim())
      .filter(Boolean),
  );

  return Array.from(unique);
}


function normalizeAnimalToken(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z]/g, '')
    .trim();
}


function normalizeDigitString(value) {
  return String(value || '').replace(/\D/g, '');
}


function normalizeDigitTokens(value) {
  return String(value || '')
    .split(/[\s,;]+/)
    .map((token) => token.replace(/\D/g, ''))
    .filter(Boolean);
}


function normalizeRotationDirection(value) {
  const normalized = String(value || '').trim().toLowerCase();

  if (!normalized) {
    return '';
  }

  if (normalized === 'counterclockwise' || normalized === 'counter-clockwise') {
    return 'anticlockwise';
  }

  if (normalized === 'clockwise' || normalized === 'anticlockwise') {
    return normalized;
  }

  return '';
}


function getSpatialRotationInstruction(question) {
  const prompt = String(question?.prompt || '').trim();
  const promptLower = prompt.toLowerCase();
  const directionFromField = normalizeRotationDirection(question?.rotation_direction);
  const directionFromPrompt = (
    promptLower.includes('anticlockwise')
    || promptLower.includes('counterclockwise')
    || promptLower.includes('counter-clockwise')
  )
    ? 'anticlockwise'
    : (promptLower.includes('clockwise') ? 'clockwise' : '');
  const direction = directionFromField || directionFromPrompt;

  const numericDegrees = Number(question?.rotation_degrees);
  const promptDegreeMatch = promptLower.match(/(\d+)\s*degrees?/);
  const promptDegrees = promptDegreeMatch ? Number.parseInt(promptDegreeMatch[1], 10) : Number.NaN;
  const degrees = Number.isFinite(numericDegrees) && numericDegrees > 0
    ? numericDegrees
    : (Number.isFinite(promptDegrees) ? promptDegrees : null);

  const fallbackPrompt = [
    'Select the option that matches the target shape',
    Number.isFinite(degrees) ? `rotated ${degrees} degrees` : 'after rotation',
    direction || '',
  ]
    .filter(Boolean)
    .join(' ')
    .trim() + '.';

  return {
    prompt: prompt || fallbackPrompt,
    direction,
    degrees,
  };
}


function arraysMatch(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => String(value) === String(right[index]));
}


function memoryInterpretation(score) {
  if (score < 9) {
    return 'Low recall range';
  }
  if (score < 17) {
    return 'Moderate recall range';
  }
  return 'Strong recall range';
}


function cognitiveInterpretation(score) {
  if (score < 12) {
    return 'High cognitive risk';
  }
  if (score < 22) {
    return 'Moderate risk';
  }
  return 'Healthy range';
}


function formatCountdown(seconds) {
  const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
}


function StepShell({ title, copy, children }) {
  return (
    <motion.div
      key={title}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -14 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-4"
    >
      <div>
        <h2 className="section-title mt-2 text-[1.45rem] sm:text-[1.65rem]">{title}</h2>
        <p className="section-copy mt-2">{copy}</p>
      </div>
      {children}
    </motion.div>
  );
}


function MatrixGrid({ grid }) {
  return (
    <div className="grid w-fit grid-cols-3 gap-1 rounded-lg border border-slate-700/80 bg-slate-950/50 p-1">
      {grid.flatMap((row, rowIndex) => row.map((cell, cellIndex) => (
        <div
          key={`${rowIndex}-${cellIndex}`}
          className={`h-7 w-7 rounded ${cell ? 'bg-cyan-300/90' : 'bg-slate-800/80'}`}
        />
      )))}
    </div>
  );
}


export default function CognitiveTestPage() {
  const navigate = useNavigate();

  const [reloadSignal, setReloadSignal] = useState(0);
  const [loadingBanks, setLoadingBanks] = useState(true);
  const [bankError, setBankError] = useState('');
  const [banks, setBanks] = useState(null);

  const [stage, setStage] = useState('intro');
  const [countdown, setCountdown] = useState(3);
  const [moduleIndex, setModuleIndex] = useState(0);
  const [startedAt, setStartedAt] = useState(null);
  const [testEndsAt, setTestEndsAt] = useState(null);
  const [sessionTimeLeft, setSessionTimeLeft] = useState(TEST_DURATION_SECONDS);
  const [timeExpired, setTimeExpired] = useState(false);
  const [orientationReference, setOrientationReference] = useState(new Date());

  const [encodingSeconds, setEncodingSeconds] = useState(10);
  const [orientation, setOrientation] = useState({ year: '', month: '', weekday: '' });
  const [digitForwardInput, setDigitForwardInput] = useState('');
  const [digitBackwardInput, setDigitBackwardInput] = useState('');
  const [reasoningChoice, setReasoningChoice] = useState('');

  const [reactionGameRunning, setReactionGameRunning] = useState(false);
  const [reactionCompleted, setReactionCompleted] = useState(false);
  const [reactionTimeLeft, setReactionTimeLeft] = useState(10);
  const [reactionTarget, setReactionTarget] = useState(null);
  const [reactionAttempts, setReactionAttempts] = useState(0);
  const [reactionHits, setReactionHits] = useState([]);
  const [reactionMisses, setReactionMisses] = useState(0);
  const [reactionTotalClicks, setReactionTotalClicks] = useState(0);

  const [patternPreviewSeconds, setPatternPreviewSeconds] = useState(3);
  const [patternPreviewVisible, setPatternPreviewVisible] = useState(true);
  const [patternSelection, setPatternSelection] = useState([]);

  const [verbalInput, setVerbalInput] = useState('');
  const [fluencySeconds, setFluencySeconds] = useState(30);
  const [fluencyLocked, setFluencyLocked] = useState(false);

  const [categoryChoice, setCategoryChoice] = useState('');
  const [stroopChoice, setStroopChoice] = useState('');
  const [symbolInput, setSymbolInput] = useState('');
  const [arithmeticAnswers, setArithmeticAnswers] = useState(['', '']);
  const [spatialChoice, setSpatialChoice] = useState('');
  const [delayedRecallInput, setDelayedRecallInput] = useState('');

  const [testConfig, setTestConfig] = useState(null);
  const [result, setResult] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [redirectCountdown, setRedirectCountdown] = useState(8);
  const [error, setError] = useState('');

  const reactionRandomRef = useRef(Math.random);
  const reactionSpawnTimerRef = useRef(null);
  const reactionMissTimerRef = useRef(null);
  const reactionRoundIdRef = useRef(0);
  const reactionTargetIdRef = useRef(0);
  const submissionStartedRef = useRef(false);

  const currentModule = MODULES[moduleIndex] || null;
  const currentModuleId = currentModule?.id || null;

  const expectedOrientation = useMemo(() => ({
    year: String(orientationReference.getFullYear()),
    month: MONTH_OPTIONS[orientationReference.getMonth()].toLowerCase(),
    weekday: WEEKDAY_OPTIONS[orientationReference.getDay()].toLowerCase(),
  }), [orientationReference]);

  const orientationScore = useMemo(() => {
    let score = 0;
    if (String(orientation.year || '').trim() === expectedOrientation.year) {
      score += 1;
    }
    if (String(orientation.month || '').trim().toLowerCase() === expectedOrientation.month) {
      score += 1;
    }
    if (String(orientation.weekday || '').trim().toLowerCase() === expectedOrientation.weekday) {
      score += 1;
    }
    return score;
  }, [orientation, expectedOrientation]);

  const digitForwardCorrect = useMemo(() => (
    testConfig
      ? normalizeDigitString(digitForwardInput) === testConfig.digitForwardSequence.join('')
      : false
  ), [digitForwardInput, testConfig]);

  const digitBackwardCorrect = useMemo(() => (
    testConfig
      ? normalizeDigitString(digitBackwardInput) === [...testConfig.digitBackwardSequence].reverse().join('')
      : false
  ), [digitBackwardInput, testConfig]);

  const reasoningCorrect = useMemo(() => (
    testConfig ? reasoningChoice === testConfig.reasoningQuestion.answer : false
  ), [reasoningChoice, testConfig]);

  const reactionAverageMs = useMemo(() => {
    if (!reactionHits.length) {
      return 0;
    }
    return Math.round(reactionHits.reduce((sum, time) => sum + time, 0) / reactionHits.length);
  }, [reactionHits]);

  const reactionAccuracyRate = useMemo(() => {
    if (!reactionTotalClicks) {
      return 0;
    }
    return Number((reactionHits.length / reactionTotalClicks).toFixed(3));
  }, [reactionHits.length, reactionTotalClicks]);

  const visualPatternCorrect = useMemo(() => {
    if (!testConfig?.visualPattern?.cells?.length) {
      return 0;
    }

    const target = new Set(testConfig.visualPattern.cells);
    const selected = new Set(patternSelection);
    let matches = 0;
    target.forEach((value) => {
      if (selected.has(value)) {
        matches += 1;
      }
    });
    return clamp(matches, 0, 3);
  }, [patternSelection, testConfig]);

  const animalLexicon = useMemo(() => (
    new Set((testConfig?.animalLexicon || []).map((word) => normalizeAnimalToken(word)).filter(Boolean))
  ), [testConfig]);

  const verbalFluencyUniqueAnimals = useMemo(() => {
    const tokens = normalizeWords(verbalInput).map((token) => normalizeAnimalToken(token));
    const unique = new Set(tokens.filter(Boolean));
    let validCount = 0;
    unique.forEach((token) => {
      if (animalLexicon.has(token)) {
        validCount += 1;
      }
    });
    return validCount;
  }, [verbalInput, animalLexicon]);

  const categoryCorrect = useMemo(() => (
    testConfig ? categoryChoice === testConfig.categoryQuestion.answer : false
  ), [categoryChoice, testConfig]);

  const stroopCorrect = useMemo(() => (
    testConfig ? stroopChoice === testConfig.stroopTrial.display_color : false
  ), [stroopChoice, testConfig]);

  const symbolDigits = useMemo(() => normalizeDigitTokens(symbolInput), [symbolInput]);

  const symbolCorrect = useMemo(() => (
    testConfig ? arraysMatch(symbolDigits, testConfig.symbolTrial.answer) : false
  ), [symbolDigits, testConfig]);

  const mentalArithmeticCorrectCount = useMemo(() => {
    if (!testConfig?.arithmeticItems) {
      return 0;
    }

    return testConfig.arithmeticItems.reduce((count, item, index) => {
      const parsed = Number.parseInt(arithmeticAnswers[index], 10);
      return count + (parsed === Number(item.answer) ? 1 : 0);
    }, 0);
  }, [testConfig, arithmeticAnswers]);

  const spatialRotationCorrect = useMemo(() => (
    testConfig ? spatialChoice === testConfig.spatialQuestion.answer : false
  ), [spatialChoice, testConfig]);

  const spatialRotationInstruction = useMemo(() => (
    testConfig?.spatialQuestion ? getSpatialRotationInstruction(testConfig.spatialQuestion) : null
  ), [testConfig]);

  const delayedRecallWords = useMemo(() => normalizeWords(delayedRecallInput), [delayedRecallInput]);

  const progressPercent = useMemo(() => {
    if (stage === 'battery') {
      return ((moduleIndex + 1) / TOTAL_MODULES) * 100;
    }
    if (stage === 'result') {
      return 100;
    }
    return 0;
  }, [stage, moduleIndex]);

  const canContinue = useMemo(() => {
    if (stage !== 'battery') {
      return false;
    }

    switch (currentModuleId) {
      case 'memoryEncoding':
        return false;
      case 'orientation':
        return Boolean(orientation.year && orientation.month && orientation.weekday);
      case 'digitForward':
        return digitForwardInput.trim().length > 0;
      case 'digitBackward':
        return digitBackwardInput.trim().length > 0;
      case 'executiveReasoning':
        return Boolean(reasoningChoice);
      case 'reactionTime':
        return reactionCompleted;
      case 'visualPattern':
        return !patternPreviewVisible && patternSelection.length > 0;
      case 'verbalFluency':
        return fluencyLocked;
      case 'categoryMatching':
        return Boolean(categoryChoice);
      case 'stroop':
        return Boolean(stroopChoice);
      case 'symbolDigit':
        return symbolInput.trim().length > 0;
      case 'mentalArithmetic':
        return arithmeticAnswers.every((value) => value.trim().length > 0);
      case 'spatialRotation':
        return Boolean(spatialChoice);
      case 'delayedRecall':
        return true;
      default:
        return false;
    }
  }, [
    stage,
    currentModuleId,
    encodingSeconds,
    orientation,
    digitForwardInput,
    digitBackwardInput,
    reasoningChoice,
    reactionCompleted,
    patternPreviewVisible,
    patternSelection.length,
    fluencyLocked,
    categoryChoice,
    stroopChoice,
    symbolInput,
    arithmeticAnswers,
    spatialChoice,
  ]);

  useEffect(() => {
    let active = true;

    async function loadBanks() {
      setLoadingBanks(true);
      setBankError('');

      try {
        const [wordBankData, reasoningData, categoryData, spatialData] = await Promise.all([
          getCognitiveWordBank(),
          getCognitiveReasoningQuestions(),
          getCognitiveCategoryQuestions(),
          getCognitiveSpatialRotationQuestions(),
        ]);

        if (!active) {
          return;
        }

        const wordBank = Array.isArray(wordBankData?.words) ? wordBankData.words : [];
        const reasoningQuestions = Array.isArray(reasoningData?.questions) ? reasoningData.questions : [];
        const stroopTrials = Array.isArray(reasoningData?.stroop_trials) ? reasoningData.stroop_trials : [];
        const symbolDigitTrials = Array.isArray(reasoningData?.symbol_digit_trials) ? reasoningData.symbol_digit_trials : [];
        const mentalArithmetic = Array.isArray(reasoningData?.mental_arithmetic) ? reasoningData.mental_arithmetic : [];
        const categoryQuestions = Array.isArray(categoryData?.questions) ? categoryData.questions : [];
        const animalLexicon = Array.isArray(categoryData?.animal_lexicon) ? categoryData.animal_lexicon : [];
        const spatialQuestions = Array.isArray(spatialData?.questions) ? spatialData.questions : [];
        const visualPatterns = Array.isArray(spatialData?.visual_patterns) ? spatialData.visual_patterns : [];

        if (
          wordBank.length < 6
          || !reasoningQuestions.length
          || !stroopTrials.length
          || !symbolDigitTrials.length
          || mentalArithmetic.length < 2
          || !categoryQuestions.length
          || !animalLexicon.length
          || !spatialQuestions.length
          || !visualPatterns.length
        ) {
          throw new Error('Cognitive banks are incomplete on the backend.');
        }

        setBanks({
          wordBank,
          reasoningQuestions,
          stroopTrials,
          symbolDigitTrials,
          mentalArithmetic,
          categoryQuestions,
          animalLexicon,
          spatialQuestions,
          visualPatterns,
        });
      } catch (loadError) {
        if (!active) {
          return;
        }
        setBankError(formatApiError(loadError, 'Unable to load curated cognitive question banks.'));
      } finally {
        if (active) {
          setLoadingBanks(false);
        }
      }
    }

    loadBanks();
    return () => {
      active = false;
    };
  }, [reloadSignal]);

  useEffect(() => {
    if (stage !== 'countdown') {
      return undefined;
    }

    if (countdown <= 0) {
      const startTime = Date.now();
      setOrientationReference(new Date());
      setStartedAt(startTime);
      setTestEndsAt(startTime + (TEST_DURATION_SECONDS * 1000));
      setSessionTimeLeft(TEST_DURATION_SECONDS);
      setTimeExpired(false);
      setStage('battery');
      setModuleIndex(0);
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setCountdown((value) => value - 1);
    }, 1000);

    return () => window.clearTimeout(timerId);
  }, [stage, countdown]);

  useEffect(() => {
    if (stage !== 'result' || !result) {
      return undefined;
    }

    if (redirectCountdown <= 0) {
      navigate('/', {
        state: {
          cognitiveTestResult: result,
        },
      });
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setRedirectCountdown((value) => value - 1);
    }, 1000);

    return () => window.clearTimeout(timerId);
  }, [stage, result, redirectCountdown, navigate]);

  useEffect(() => {
    if (stage !== 'battery' || !testEndsAt || result) {
      return undefined;
    }

    const syncSessionTimeLeft = () => {
      const remaining = Math.max(0, Math.ceil((testEndsAt - Date.now()) / 1000));
      setSessionTimeLeft((current) => (current === remaining ? current : remaining));
    };

    syncSessionTimeLeft();
    const timerId = window.setInterval(syncSessionTimeLeft, 250);

    return () => window.clearInterval(timerId);
  }, [stage, testEndsAt, result]);

  useEffect(() => {
    if (
      stage !== 'battery'
      || !startedAt
      || result
      || sessionTimeLeft > 0
      || timeExpired
      || submissionStartedRef.current
    ) {
      return;
    }

    setTimeExpired(true);
    void submitBattery({ forceDurationSeconds: TEST_DURATION_SECONDS });
  }, [stage, startedAt, result, sessionTimeLeft, timeExpired]);

  useEffect(() => {
    if (stage !== 'battery') {
      return;
    }

    if (currentModuleId === 'memoryEncoding') {
      setEncodingSeconds(10);
    }

    if (currentModuleId === 'reactionTime') {
      reactionRoundIdRef.current += 1;
      clearReactionTimers();
      setReactionGameRunning(false);
      setReactionCompleted(false);
      setReactionTimeLeft(10);
      setReactionTarget(null);
      setReactionAttempts(0);
      setReactionHits([]);
      setReactionMisses(0);
      setReactionTotalClicks(0);
    }

    if (currentModuleId === 'visualPattern') {
      setPatternPreviewSeconds(3);
      setPatternPreviewVisible(true);
      setPatternSelection([]);
    }

    if (currentModuleId === 'verbalFluency') {
      setFluencySeconds(30);
      setFluencyLocked(false);
      setVerbalInput('');
    }
  }, [stage, currentModuleId]);

  useEffect(() => {
    if (stage !== 'battery' || currentModuleId !== 'memoryEncoding' || encodingSeconds <= 0) {
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setEncodingSeconds((value) => value - 1);
    }, 1000);

    return () => window.clearTimeout(timerId);
  }, [stage, currentModuleId, encodingSeconds]);

  useEffect(() => {
    if (stage !== 'battery' || currentModuleId !== 'memoryEncoding' || encodingSeconds > 0) {
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setModuleIndex((value) => {
        if (value >= TOTAL_MODULES - 1) {
          return value;
        }
        return value + 1;
      });
    }, 150);

    return () => window.clearTimeout(timerId);
  }, [stage, currentModuleId, encodingSeconds]);

  useEffect(() => {
    if (stage !== 'battery' || currentModuleId !== 'visualPattern' || !patternPreviewVisible) {
      return undefined;
    }

    if (patternPreviewSeconds <= 0) {
      setPatternPreviewVisible(false);
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setPatternPreviewSeconds((value) => value - 1);
    }, 1000);

    return () => window.clearTimeout(timerId);
  }, [stage, currentModuleId, patternPreviewVisible, patternPreviewSeconds]);

  useEffect(() => {
    if (stage !== 'battery' || currentModuleId !== 'verbalFluency' || fluencyLocked) {
      return undefined;
    }

    if (fluencySeconds <= 0) {
      setFluencyLocked(true);
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setFluencySeconds((value) => value - 1);
    }, 1000);

    return () => window.clearTimeout(timerId);
  }, [stage, currentModuleId, fluencyLocked, fluencySeconds]);

  useEffect(() => {
    if (stage !== 'battery' || currentModuleId !== 'reactionTime' || !reactionGameRunning) {
      return undefined;
    }

    const maxAttemptsReached = reactionAttempts >= MAX_REACTION_ATTEMPTS && !reactionTarget;
    if (reactionTimeLeft <= 0 || maxAttemptsReached) {
      clearReactionTimers();
      setReactionGameRunning(false);
      setReactionCompleted(true);
      setReactionTarget(null);
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setReactionTimeLeft((value) => value - 1);
    }, 1000);

    return () => window.clearTimeout(timerId);
  }, [stage, currentModuleId, reactionGameRunning, reactionTimeLeft, reactionAttempts, reactionTarget]);

  useEffect(() => {
    if (stage !== 'battery' || currentModuleId !== 'reactionTime' || !reactionGameRunning || reactionTarget) {
      return undefined;
    }

    if (reactionAttempts >= MAX_REACTION_ATTEMPTS) {
      clearReactionTimers();
      setReactionGameRunning(false);
      setReactionCompleted(true);
      return undefined;
    }

    const delay = 250 + Math.floor(reactionRandomRef.current() * 650);
    const roundId = reactionRoundIdRef.current;

    reactionSpawnTimerRef.current = window.setTimeout(() => {
      if (reactionRoundIdRef.current !== roundId) {
        return;
      }

      const targetId = reactionTargetIdRef.current + 1;
      reactionTargetIdRef.current = targetId;

      setReactionAttempts((value) => value + 1);
      setReactionTarget({
        id: targetId,
        x: 10 + (reactionRandomRef.current() * 80),
        y: 15 + (reactionRandomRef.current() * 70),
        spawnedAt: performance.now(),
      });

      reactionMissTimerRef.current = window.setTimeout(() => {
        if (reactionRoundIdRef.current !== roundId) {
          return;
        }

        setReactionTarget((current) => {
          if (!current || current.id !== targetId) {
            return current;
          }
          setReactionMisses((value) => value + 1);
          return null;
        });
      }, 1000);
    }, delay);

    return () => {
      if (reactionSpawnTimerRef.current) {
        window.clearTimeout(reactionSpawnTimerRef.current);
        reactionSpawnTimerRef.current = null;
      }
    };
  }, [stage, currentModuleId, reactionGameRunning, reactionTarget, reactionAttempts]);

  useEffect(() => () => {
    clearReactionTimers();
  }, []);

  function clearReactionTimers() {
    if (reactionSpawnTimerRef.current) {
      window.clearTimeout(reactionSpawnTimerRef.current);
      reactionSpawnTimerRef.current = null;
    }

    if (reactionMissTimerRef.current) {
      window.clearTimeout(reactionMissTimerRef.current);
      reactionMissTimerRef.current = null;
    }
  }

  function resetResponses() {
    submissionStartedRef.current = false;
    reactionRoundIdRef.current += 1;
    setEncodingSeconds(10);
    setStartedAt(null);
    setTestEndsAt(null);
    setSessionTimeLeft(TEST_DURATION_SECONDS);
    setTimeExpired(false);
    setOrientation({ year: '', month: '', weekday: '' });
    setDigitForwardInput('');
    setDigitBackwardInput('');
    setReasoningChoice('');

    clearReactionTimers();
    setReactionGameRunning(false);
    setReactionCompleted(false);
    setReactionTimeLeft(10);
    setReactionTarget(null);
    setReactionAttempts(0);
    setReactionHits([]);
    setReactionMisses(0);
    setReactionTotalClicks(0);

    setPatternPreviewSeconds(3);
    setPatternPreviewVisible(true);
    setPatternSelection([]);

    setVerbalInput('');
    setFluencySeconds(30);
    setFluencyLocked(false);

    setCategoryChoice('');
    setStroopChoice('');
    setSymbolInput('');
    setArithmeticAnswers(['', '']);
    setSpatialChoice('');
    setDelayedRecallInput('');

    setResult(null);
    setError('');
  }

  function startBattery() {
    if (!banks || loadingBanks) {
      return;
    }

    const seed = Date.now();
    const random = createSeededRandom(seed);
    reactionRandomRef.current = random;

    const encodingWords = pickRandom(banks.wordBank, 6, random).map((word) => String(word).trim().toLowerCase());
    const reasoningQuestion = pickRandom(banks.reasoningQuestions, 1, random)[0];
    const categoryQuestion = pickRandom(banks.categoryQuestions, 1, random)[0];
    const spatialQuestion = pickRandom(banks.spatialQuestions, 1, random)[0];
    const visualPattern = pickRandom(banks.visualPatterns, 1, random)[0];
    const stroopTrial = pickRandom(banks.stroopTrials, 1, random)[0];
    const symbolTrial = pickRandom(banks.symbolDigitTrials, 1, random)[0];
    const arithmeticItems = pickRandom(banks.mentalArithmetic, 2, random);

    setTestConfig({
      seed,
      encodingWords,
      reasoningQuestion,
      categoryQuestion,
      spatialQuestion,
      visualPattern,
      stroopTrial,
      symbolTrial,
      arithmeticItems,
      animalLexicon: banks.animalLexicon,
      digitForwardSequence: buildDigitSequence(5, random),
      digitBackwardSequence: buildDigitSequence(4, random),
    });

    resetResponses();
    setRedirectCountdown(8);
    setCountdown(3);
    setStage('countdown');
  }

  function startReactionGame() {
    reactionRoundIdRef.current += 1;
    clearReactionTimers();
    setReactionGameRunning(true);
    setReactionCompleted(false);
    setReactionTimeLeft(10);
    setReactionTarget(null);
    setReactionAttempts(0);
    setReactionHits([]);
    setReactionMisses(0);
    setReactionTotalClicks(0);
  }

  function handleMissClick(e) {
    if (!reactionGameRunning || !reactionTarget) return;
    
    // Only count miss if there's still a target (it wasn't already hit)
    setReactionTotalClicks((prev) => prev + 1);
    setReactionMisses((prev) => prev + 1);
  }

  function handleReactionHit(e) {
    if (e) {
      e.stopPropagation();
    }
    if (!reactionGameRunning || !reactionTarget) return;
    
    // Immediately clear the target to prevent multiple clicks on same target
    const currentTarget = reactionTarget;
    setReactionTarget(null);
    
    // Now increment counters
    setReactionTotalClicks((prev) => prev + 1);

    if (reactionMissTimerRef.current) {
      window.clearTimeout(reactionMissTimerRef.current);
      reactionMissTimerRef.current = null;
    }

    const elapsed = Math.max(0, Math.round(performance.now() - currentTarget.spawnedAt));
    setReactionHits((times) => [...times, elapsed]);
  }

  function togglePatternCell(index) {
    if (patternPreviewVisible) {
      return;
    }

    setPatternSelection((current) => {
      if (current.includes(index)) {
        return current.filter((value) => value !== index);
      }

      if (current.length >= 6) {
        return current;
      }

      return [...current, index];
    });
  }

  function updateArithmeticAnswer(index, value) {
    setArithmeticAnswers((current) => {
      const next = [...current];
      next[index] = value;
      return next;
    });
  }

  async function submitBattery(options = {}) {
    const { forceDurationSeconds } = options;

    if (!testConfig || !startedAt || submissionStartedRef.current) {
      return;
    }

    submissionStartedRef.current = true;
    clearReactionTimers();
    setReactionGameRunning(false);
    setReactionCompleted(true);
    setReactionTarget(null);
    setSubmitting(true);
    setError('');
    let submissionSucceeded = false;

    try {
      const totalDurationSeconds = Number.isFinite(forceDurationSeconds)
        ? forceDurationSeconds
        : Math.max(0, Math.round((Date.now() - startedAt) / 1000));

      const payload = {
        total_duration_seconds: totalDurationSeconds,
        orientation_score: orientationScore,
        digit_span_forward_correct: digitForwardCorrect,
        digit_span_backward_correct: digitBackwardCorrect,
        executive_reasoning_correct: reasoningCorrect,
        reaction_average_ms: reactionAverageMs,
        reaction_accuracy_rate: reactionAccuracyRate,
        reaction_missed_targets: reactionMisses,
        visual_pattern_correct: visualPatternCorrect,
        verbal_fluency_unique_animals: verbalFluencyUniqueAnimals,
        category_matching_correct: categoryCorrect,
        stroop_correct: stroopCorrect,
        symbol_digit_correct: symbolCorrect,
        mental_arithmetic_correct_count: mentalArithmeticCorrectCount,
        spatial_rotation_correct: spatialRotationCorrect,
        delayed_recall_words: delayedRecallWords,
        encoded_words: testConfig.encodingWords,
      };

      const scores = await submitCognitiveTestResults(payload);
      setResult({
        memory_score: Number(scores.memory_score ?? 0),
        cognitive_score: Number(scores.cognitive_score ?? 0),
        reliability_flag: scores.reliability_flag || 'standard',
        reliability_message: scores.reliability_message || '',
        duration_minutes: Number(scores.duration_minutes ?? 0),
        module_scores: scores.module_scores || {},
      });
      submissionSucceeded = true;
      setStage('result');
    } catch (submitError) {
      setError(formatApiError(submitError, 'Unable to submit cognitive test results.'));
    } finally {
      setSubmitting(false);
      if (!submissionSucceeded) {
        submissionStartedRef.current = false;
      }
    }
  }

  async function handleContinue() {
    if (!canContinue || stage !== 'battery' || submitting || sessionTimeLeft <= 0) {
      return;
    }

    if (moduleIndex >= TOTAL_MODULES - 1) {
      return;
    }

    setModuleIndex((value) => value + 1);
  }

  function handleSubmitFinal() {
    if (
      stage !== 'battery'
      || submitting
      || sessionTimeLeft <= 0
      || moduleIndex < TOTAL_MODULES - 1
      || !testConfig
      || !startedAt
    ) {
      return;
    }

    void submitBattery();
  }

  function finishNow() {
    if (!result) {
      return;
    }

    navigate('/', {
      state: {
        cognitiveTestResult: result,
      },
    });
  }

  function renderIntro() {
    return (
      <StepShell
        title="Cognitive Test Modules"
        copy="This test includes 14 short tasks that evaluate different cognitive abilities."
      >
        <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4 text-sm text-slate-200">
          <p>
            This test takes about <span className="font-semibold text-cyan-200">10 minutes</span>.
          </p>
          <p className="mt-2">
            You will complete a series of small tasks that measure how your brain processes information.
          </p>
          <p className="mt-2">A session timer counts down from 10:00, and your scores are calculated automatically when it ends.</p>
          <p className="mt-2">Please try to focus and answer honestly.</p>
        </div>

        <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4 text-sm text-slate-200">
          <p className="font-semibold text-slate-100">They measure:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>memory</li>
            <li>attention</li>
            <li>reaction speed</li>
            <li>reasoning</li>
            <li>language</li>
            <li>visual processing</li>
          </ul>
        </div>

        <div className="grid gap-3 rounded-2xl border border-slate-700/70 bg-slate-950/50 p-4 sm:grid-cols-2">
          {MODULES.map((module) => (
            <div key={module.id} className="rounded-xl border border-slate-700/60 bg-slate-900/50 px-3 py-2 text-sm text-slate-200">
              {module.title}
            </div>
          ))}
        </div>

        {bankError ? (
          <p className="rounded-xl border border-rose-400/40 bg-rose-500/15 px-4 py-3 text-sm text-rose-100">{bankError}</p>
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-slate-400">
            The test usually takes <span className="font-semibold">8-10 minutes</span>. If it finishes much faster than expected, the results may not be accurate.
          </p>
          <div className="flex items-center gap-2">
            <p className="text-xs text-slate-300">
              When you're ready, click <span className="font-semibold text-cyan-200">Start Test</span>. The tasks will appear one by one.
            </p>
            <button
              type="button"
              onClick={() => setReloadSignal((value) => value + 1)}
              className="ghost-button"
              disabled={loadingBanks}
            >
              Reload Questions
            </button>
            <button
              type="button"
              onClick={startBattery}
              disabled={loadingBanks || Boolean(bankError) || !banks}
              className="neon-button disabled:cursor-not-allowed disabled:opacity-45"
            >
              {loadingBanks ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading Banks...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Start Test
                </>
              )}
            </button>
          </div>
        </div>
      </StepShell>
    );
  }

  function renderCountdown() {
    return (
      <StepShell
        title="Get Ready"
        copy="The test starts now. Stay focused and complete each task one by one."
      >
        <div className="flex flex-col items-center justify-center rounded-2xl border border-cyan-400/35 bg-cyan-500/10 py-12">
          <p className="text-sm uppercase tracking-wide text-cyan-200">Starting In</p>
          <motion.p
            key={countdown}
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-3 text-6xl font-bold text-cyan-100"
          >
            {countdown}
          </motion.p>
        </div>
      </StepShell>
    );
  }

  function renderBattery() {
    if (!currentModule || !testConfig) {
      return null;
    }

    switch (currentModuleId) {
      case 'memoryEncoding':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="grid gap-3 sm:grid-cols-3">
              {testConfig.encodingWords.map((word) => (
                <div key={word} className="rounded-xl border border-slate-700/70 bg-slate-950/45 px-4 py-3 text-center text-sm font-semibold text-cyan-100">
                  {word}
                </div>
              ))}
            </div>
            <p className="rounded-xl border border-slate-700/70 bg-slate-950/50 px-4 py-3 text-sm text-slate-300">
              This module auto-continues when the memorization window ends. Time remaining: <span className="font-semibold text-cyan-200">{encodingSeconds}s</span>
            </p>
          </StepShell>
        );

      case 'orientation':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="grid gap-4 sm:grid-cols-3">
              <label className="block">
                <span className="metric-label">Year</span>
                <input
                  value={orientation.year}
                  onChange={(event) => setOrientation((current) => ({ ...current, year: event.target.value }))}
                  type="number"
                  placeholder="2026"
                  className="field-input mt-2"
                />
              </label>
              <label className="block">
                <span className="metric-label">Month</span>
                <select
                  value={orientation.month}
                  onChange={(event) => setOrientation((current) => ({ ...current, month: event.target.value }))}
                  className="field-input field-select-themed mt-2"
                >
                  <option value="">Select month</option>
                  {MONTH_OPTIONS.map((month) => (
                    <option key={month} value={month}>{month}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="metric-label">Weekday</span>
                <select
                  value={orientation.weekday}
                  onChange={(event) => setOrientation((current) => ({ ...current, weekday: event.target.value }))}
                  className="field-input field-select-themed mt-2"
                >
                  <option value="">Select day</option>
                  {WEEKDAY_OPTIONS.map((weekday) => (
                    <option key={weekday} value={weekday}>{weekday}</option>
                  ))}
                </select>
              </label>
            </div>
          </StepShell>
        );

      case 'digitForward':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 px-4 py-6 text-center">
              <p className="metric-label">Digit sequence</p>
              <p className="mt-2 text-3xl font-semibold tracking-[0.32em] text-white">{testConfig.digitForwardSequence.join(' ')}</p>
            </div>
            <label className="block">
              <span className="metric-label">Type the same sequence</span>
              <input
                value={digitForwardInput}
                onChange={(event) => setDigitForwardInput(event.target.value)}
                type="text"
                placeholder="Example: 34179"
                className="field-input mt-2"
              />
            </label>
          </StepShell>
        );

      case 'digitBackward':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 px-4 py-6 text-center">
              <p className="metric-label">Digit sequence</p>
              <p className="mt-2 text-3xl font-semibold tracking-[0.32em] text-white">{testConfig.digitBackwardSequence.join(' ')}</p>
            </div>
            <label className="block">
              <span className="metric-label">Type in reverse order</span>
              <input
                value={digitBackwardInput}
                onChange={(event) => setDigitBackwardInput(event.target.value)}
                type="text"
                placeholder="Example: 9146"
                className="field-input mt-2"
              />
            </label>
          </StepShell>
        );

      case 'executiveReasoning':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <p className="rounded-xl border border-slate-700/70 bg-slate-950/50 px-4 py-3 text-sm text-slate-100">
              {testConfig.reasoningQuestion.prompt}
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              {testConfig.reasoningQuestion.options.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setReasoningChoice(option)}
                  className={`rounded-2xl border px-4 py-3 text-left text-sm font-semibold transition ${
                    reasoningChoice === option
                      ? 'border-cyan-300 bg-cyan-400/15 text-cyan-100'
                      : 'border-slate-700/70 bg-slate-950/45 text-slate-200 hover:border-slate-500'
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </StepShell>
        );

      case 'reactionTime':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="grid gap-3 grid-cols-2 md:grid-cols-3">
              <div className="rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2">
                <p className="metric-label">Time Left</p>
                <p className="mt-1 text-lg font-semibold text-white">{reactionTimeLeft}s</p>
              </div>
              <div className="rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2">
                <p className="metric-label">Targets Spawned</p>
                <p className="mt-1 text-lg font-semibold text-white">{reactionAttempts} / {MAX_REACTION_ATTEMPTS}</p>
              </div>
              <div className="rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2">
                <p className="metric-label">Avg. Reaction</p>
                <p className="mt-1 text-lg font-semibold text-white">{reactionAverageMs || 0} ms</p>
              </div>
              <div className="rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2">
                <p className="metric-label">Hits / Misses</p>
                <p className="mt-1 text-lg font-semibold text-white">
                  <span className="text-emerald-400">{reactionHits.length}</span> / <span className="text-rose-400">{reactionMisses}</span>
                </p>
              </div>
              <div className="rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2">
                <p className="metric-label">Total Clicks</p>
                <p className="mt-1 text-lg font-semibold text-white">{reactionTotalClicks}</p>
              </div>
              <div className="rounded-xl border border-slate-700/70 bg-slate-950/45 px-3 py-2">
                <p className="metric-label">Accuracy</p>
                <p className="mt-1 text-lg font-semibold text-white">{(reactionAccuracyRate * 100).toFixed(0)}%</p>
              </div>
            </div>

            {!reactionGameRunning && !reactionCompleted ? (
              <button type="button" onClick={startReactionGame} className="neon-button">
                <Target className="h-4 w-4" />
                Start Reaction Round
              </button>
            ) : null}

            <div 
              className="relative h-56 rounded-2xl border border-slate-700/70 bg-slate-950/45 overflow-hidden cursor-crosshair"
              onClick={reactionGameRunning ? handleMissClick : undefined}
            >
              {reactionTarget ? (
                <button
                  type="button"
                  onClick={handleReactionHit}
                  className="absolute h-12 w-12 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-300 shadow-[0_0_25px_rgba(34,211,238,0.55)] transition hover:scale-105"
                  style={{ left: `${reactionTarget.x}%`, top: `${reactionTarget.y}%` }}
                  aria-label="Tap target"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-400">
                  {reactionGameRunning ? 'Stay alert. Next target is about to appear...' : 'Start the round to begin.'}
                </div>
              )}
            </div>

            {reactionCompleted ? (
              <p className="rounded-xl border border-cyan-400/30 bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100">
                Round complete. Hits: {reactionHits.length}, misses: {reactionMisses}, total clicks: {reactionTotalClicks}, accuracy: {(reactionAccuracyRate * 100).toFixed(0)}%.
              </p>
            ) : null}
          </StepShell>
        );

      case 'visualPattern':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="grid gap-5 md:grid-cols-2">
              <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4">
                <p className="metric-label">Reference Grid</p>
                <div className="mt-3 grid w-fit grid-cols-3 gap-2">
                  {Array.from({ length: 9 }, (_, index) => {
                    const isActive = patternPreviewVisible && testConfig.visualPattern.cells.includes(index);
                    return (
                      <div
                        key={`reference-${index}`}
                        className={`h-10 w-10 rounded ${isActive ? 'bg-cyan-300/90' : 'bg-slate-800/80'}`}
                      />
                    );
                  })}
                </div>
                <p className="mt-3 text-xs text-slate-300">
                  {patternPreviewVisible ? `Pattern visible for ${patternPreviewSeconds}s` : 'Pattern hidden. Rebuild it on the response grid.'}
                </p>
              </div>

              <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4">
                <p className="metric-label">Response Grid</p>
                <div className="mt-3 grid w-fit grid-cols-3 gap-2">
                  {Array.from({ length: 9 }, (_, index) => {
                    const selected = patternSelection.includes(index);
                    return (
                      <button
                        key={`response-${index}`}
                        type="button"
                        onClick={() => togglePatternCell(index)}
                        className={`h-10 w-10 rounded border transition ${
                          selected
                            ? 'border-cyan-200 bg-cyan-400/60'
                            : 'border-slate-600 bg-slate-800/80 hover:border-slate-400'
                        }`}
                        disabled={patternPreviewVisible}
                        aria-label={`Cell ${index + 1}`}
                      />
                    );
                  })}
                </div>
                <p className="mt-3 text-xs text-slate-300">
                  Correct cells count toward score, up to 3 points.
                </p>
              </div>
            </div>
          </StepShell>
        );

      case 'verbalFluency':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <p className="rounded-xl border border-slate-700/70 bg-slate-950/50 px-4 py-3 text-sm text-slate-200">
              Timer: <span className="font-semibold text-cyan-200">{fluencySeconds}s</span>. Enter one animal per word, separated by spaces or commas.
            </p>
            <textarea
              value={verbalInput}
              onChange={(event) => setVerbalInput(event.target.value)}
              rows={6}
              placeholder="dog, tiger, whale, eagle..."
              className="field-input resize-none"
              disabled={fluencyLocked}
            />
            {fluencyLocked ? (
              <p className="text-sm text-cyan-100">Time complete. Valid unique animals detected: {verbalFluencyUniqueAnimals}.</p>
            ) : null}
          </StepShell>
        );

      case 'categoryMatching':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <p className="rounded-xl border border-slate-700/70 bg-slate-950/50 px-4 py-3 text-sm text-slate-100">
              {testConfig.categoryQuestion.prompt}
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              {testConfig.categoryQuestion.options.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setCategoryChoice(option)}
                  className={`rounded-2xl border px-4 py-3 text-left text-sm font-semibold transition ${
                    categoryChoice === option
                      ? 'border-cyan-300 bg-cyan-400/15 text-cyan-100'
                      : 'border-slate-700/70 bg-slate-950/45 text-slate-200 hover:border-slate-500'
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </StepShell>
        );

      case 'stroop':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 px-4 py-6 text-center">
              <p className="metric-label">Select the display color</p>
              <p className={`mt-3 text-4xl font-bold tracking-[0.2em] ${STROOP_COLOR_CLASSES[testConfig.stroopTrial.display_color] || 'text-white'}`}>
                {testConfig.stroopTrial.word}
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {testConfig.stroopTrial.options.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setStroopChoice(option)}
                  className={`rounded-2xl border px-4 py-3 text-left text-sm font-semibold uppercase transition ${
                    stroopChoice === option
                      ? 'border-cyan-300 bg-cyan-400/15 text-cyan-100'
                      : 'border-slate-700/70 bg-slate-950/45 text-slate-200 hover:border-slate-500'
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </StepShell>
        );

      case 'symbolDigit':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4">
              <p className="metric-label">Mapping Key</p>
              <div className="mt-3 grid gap-2 sm:grid-cols-4">
                {Object.entries(testConfig.symbolTrial.mapping).map(([symbol, value]) => (
                  <div key={symbol} className="rounded-lg border border-slate-700/70 bg-slate-900/50 px-3 py-2 text-center text-sm text-slate-100">
                    <span className="font-semibold capitalize">{symbol}</span>
                    <span className="mx-2 text-slate-500">-</span>
                    <span className="font-bold text-cyan-200">{value}</span>
                  </div>
                ))}
              </div>
            </div>

            <p className="rounded-xl border border-slate-700/70 bg-slate-950/50 px-4 py-3 text-sm text-slate-100">
              Sequence: <span className="font-semibold text-cyan-200">{testConfig.symbolTrial.sequence.join(' | ')}</span>
            </p>

            <label className="block">
              <span className="metric-label">Type digits in order (space-separated)</span>
              <input
                value={symbolInput}
                onChange={(event) => setSymbolInput(event.target.value)}
                type="text"
                placeholder="1 2 3 2"
                className="field-input mt-2"
              />
            </label>
          </StepShell>
        );

      case 'mentalArithmetic':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="grid gap-4">
              {testConfig.arithmeticItems.map((item, index) => (
                <label key={item.id} className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4">
                  <span className="metric-label">Question {index + 1}</span>
                  <p className="mt-2 text-lg font-semibold text-white">{item.prompt}</p>
                  <input
                    value={arithmeticAnswers[index]}
                    onChange={(event) => updateArithmeticAnswer(index, event.target.value)}
                    type="number"
                    className="field-input mt-3"
                    placeholder="Enter answer"
                  />
                </label>
              ))}
            </div>
          </StepShell>
        );

      case 'spatialRotation':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <div className="rounded-xl border border-slate-700/70 bg-slate-950/50 px-4 py-3 text-sm text-slate-100">
              <p>
                {spatialRotationInstruction?.prompt || testConfig.spatialQuestion.prompt}
              </p>
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                {Number.isFinite(spatialRotationInstruction?.degrees) ? (
                  <span className="rounded-full border border-slate-500/70 bg-slate-900/70 px-2 py-1 text-slate-100">
                    Rotation: {spatialRotationInstruction.degrees} deg
                  </span>
                ) : null}
                {spatialRotationInstruction?.direction ? (
                  <span className="rounded-full border border-cyan-400/40 bg-cyan-500/10 px-2 py-1 font-semibold capitalize text-cyan-100">
                    Direction: {spatialRotationInstruction.direction}
                  </span>
                ) : null}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4">
              <p className="metric-label">Target Shape</p>
              <div className="mt-3">
                <MatrixGrid grid={testConfig.spatialQuestion.target} />
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              {testConfig.spatialQuestion.options.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setSpatialChoice(option.id)}
                  className={`rounded-2xl border p-3 text-left transition ${
                    spatialChoice === option.id
                      ? 'border-cyan-300 bg-cyan-400/15'
                      : 'border-slate-700/70 bg-slate-950/45 hover:border-slate-500'
                  }`}
                >
                  <p className="metric-label">Option {option.id}</p>
                  <div className="mt-2">
                    <MatrixGrid grid={option.grid} />
                  </div>
                </button>
              ))}
            </div>
          </StepShell>
        );

      case 'delayedRecall':
        return (
          <StepShell title={currentModule.title} copy={currentModule.copy}>
            <textarea
              value={delayedRecallInput}
              onChange={(event) => setDelayedRecallInput(event.target.value)}
              rows={5}
              placeholder="Enter recalled words separated by spaces or commas."
              className="field-input resize-none"
            />
            <p className="text-sm text-slate-400">
              Memory score = (correct recalled words / 6) x 25.
            </p>
            {sessionTimeLeft > 0 ? (
              <p className="rounded-xl border border-slate-700/70 bg-slate-950/50 px-4 py-3 text-sm text-slate-300">
                Final module: submit when ready, or wait for <span className="font-semibold text-cyan-200">00:00</span> for automatic submission.
              </p>
            ) : null}
          </StepShell>
        );

      default:
        return null;
    }
  }

  function renderResult() {
    if (!result) {
      return null;
    }

    return (
      <StepShell
        title="Test Complete"
        copy="Your scores are ready and can now be added to the dashboard."
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4">
            <p className="metric-label">Memory Score</p>
            <p className="mt-2 text-2xl font-semibold text-white">{result.memory_score} / 25</p>
            <p className="mt-2 text-sm text-slate-300">Interpretation: {memoryInterpretation(result.memory_score)}</p>
          </div>

          <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4">
            <p className="metric-label">Cognitive Score</p>
            <p className="mt-2 text-2xl font-semibold text-white">{result.cognitive_score} / 30</p>
            <p className="mt-2 text-sm text-slate-300">Interpretation: {cognitiveInterpretation(result.cognitive_score)}</p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-700/70 bg-slate-950/45 p-4 text-sm text-slate-200">
          <p className="font-semibold text-white">Reliability: {String(result.reliability_flag || 'standard').toUpperCase()}</p>
          <p className="mt-2">{result.reliability_message || 'No reliability note available.'}</p>
          <p className="mt-2">Duration: {result.duration_minutes || 0} minutes</p>
        </div>

        <p className="rounded-2xl border border-cyan-400/30 bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100">
          Redirecting to dashboard in {redirectCountdown}s. Memory and Cognitive scores will auto-fill in the risk form.
        </p>

        <div className="flex justify-end">
          <button type="button" onClick={finishNow} className="neon-button">
            Return to Dashboard
          </button>
        </div>
      </StepShell>
    );
  }

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-ann-bg text-slate-100">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <motion.div
          animate={{ x: [0, 16, 0], y: [0, -10, 0] }}
          transition={{ duration: 11, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute left-[-7rem] top-[-5rem] h-72 w-72 rounded-full bg-ann-indigo/25 blur-3xl"
        />
        <motion.div
          animate={{ x: [0, -14, 0], y: [0, 12, 0] }}
          transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute right-[-7rem] top-[18%] h-72 w-72 rounded-full bg-ann-cyan/20 blur-3xl"
        />
      </div>

      <main className="mx-auto w-full max-w-5xl px-4 pb-16 pt-10 sm:px-6 lg:px-8">
        <section className="glass-card p-6 sm:p-8">
          <div className="relative z-10 space-y-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="section-kicker">Cognitive Screening</p>
                <h1 className="section-title mt-2">10-Minute Cognitive Test</h1>
                <p className="section-copy mt-2">
                  This short test checks different thinking abilities like memory, attention, reaction speed, and problem solving.
                </p>
              </div>
              <button type="button" onClick={() => navigate('/')} className="ghost-button">
                <ArrowLeft className="h-4 w-4" />
                Back to Dashboard
              </button>
            </div>

            <div className="surface-panel">
              <div className="mb-3 flex items-center justify-between gap-3">
                <p className="metric-label">
                  {stage === 'battery' ? `Step ${moduleIndex + 1} / ${TOTAL_MODULES}` : stage === 'result' ? `Step ${TOTAL_MODULES} / ${TOTAL_MODULES}` : 'Before You Begin'}
                </p>
                <div className="inline-flex items-center gap-2 text-xs text-slate-300">
                  <Clock3 className="h-4 w-4 text-ann-cyan" />
                  {stage === 'countdown'
                    ? `Starting in ${countdown}s`
                    : stage === 'battery'
                      ? `Time left: ${formatCountdown(sessionTimeLeft)}`
                      : stage === 'result'
                        ? 'Scores ready'
                        : '10:00 guided session'}
                </div>
              </div>
              <div className="h-2 rounded-full bg-slate-800">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${progressPercent}%` }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                  className="h-full rounded-full bg-gradient-to-r from-ann-indigo to-ann-cyan"
                />
              </div>
            </div>

            <AnimatePresence mode="wait">
              {stage === 'intro' ? renderIntro() : null}
              {stage === 'countdown' ? renderCountdown() : null}
              {stage === 'battery' ? renderBattery() : null}
              {stage === 'result' ? renderResult() : null}
            </AnimatePresence>

            {stage === 'battery' ? (
              <p className="surface-panel px-4 py-3 text-sm text-slate-200">
                {sessionTimeLeft > 0
                  ? `Time left: ${formatCountdown(sessionTimeLeft)}. ${moduleIndex >= TOTAL_MODULES - 1
                    ? 'Final module reached. Submit when ready, or auto-submit runs at 00:00.'
                    : 'Use Continue to move to the next module.'}`
                  : 'Time is up. Calculating your scores now...'}
              </p>
            ) : null}

            {error ? (
              <p className="rounded-xl border border-rose-400/40 bg-rose-500/15 px-4 py-3 text-sm text-rose-100">{error}</p>
            ) : null}

            {stage === 'battery' ? (
              <div className="flex flex-wrap justify-end gap-3">
                {currentModuleId !== 'memoryEncoding' && sessionTimeLeft > 0 && moduleIndex < TOTAL_MODULES - 1 ? (
                  <button
                    type="button"
                    onClick={handleContinue}
                    disabled={submitting || !canContinue}
                    className="ghost-button disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    Continue
                  </button>
                ) : null}

                {sessionTimeLeft > 0 && moduleIndex >= TOTAL_MODULES - 1 ? (
                  <button
                    type="button"
                    onClick={handleSubmitFinal}
                    disabled={submitting || !testConfig || !startedAt || !canContinue}
                    className="neon-button disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    {submitting ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Calculating Scores...
                      </>
                    ) : (
                      <>
                        <BrainCircuit className="h-4 w-4" />
                        Submit Test
                      </>
                    )}
                  </button>
                ) : null}

                {sessionTimeLeft <= 0 ? (
                  <button
                    type="button"
                    onClick={() => submitBattery({ forceDurationSeconds: TEST_DURATION_SECONDS })}
                    disabled={submitting || !testConfig || !startedAt}
                    className="neon-button disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    {submitting ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Calculating Scores...
                      </>
                    ) : (
                      <>
                        <BrainCircuit className="h-4 w-4" />
                        Retry Calculation
                      </>
                    )}
                  </button>
                ) : null}
              </div>
            ) : null}

            <p className="border-t border-slate-700/70 pt-4 text-xs leading-6 text-slate-400">
              This test is inspired by commonly used cognitive assessments and is intended for learning and research purposes only. It is not a medical diagnosis.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
