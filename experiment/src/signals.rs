use std::range::Range;
use crate::signal_optimization::{SignalParams, SignalParamsDir};

pub fn signal_unknown_09_03() -> SignalParams {
    SignalParams {
        hold_ms: 250,
        up: Some(SignalParamsDir { threshold: -0.15453738797718342, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 1.1629761023166723, derivative2_weight: 0.32481462380662857, imbalance1_weights: [-0.09802006258516288, -0.04839597102546962, -0.0549509873464885, -0.15984030244392058,0.0], imbalance2_weights: [-0.06759615045675671, -0.049698124522988454, 0.061509796401540175, 0.06838655861917534, 0.0] }),
        down: Some(SignalParamsDir { threshold: -0.12809776356042107, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 1.2524761056539522, derivative2_weight: 1.0711123993550387, imbalance1_weights: [-0.08348685530090004, -0.19510914374521984, -0.22190776923813244, -0.004042762088653724,0.0], imbalance2_weights: [-0.05784155826200306, -0.01112889975677297, -0.06887535463793135, -0.12454322697759054, 0.0] })
    }
}

pub fn signal_spearman_09_03() -> SignalParams {
    SignalParams {
        hold_ms: 250,
        up: Some(SignalParamsDir { threshold: 50.906435140756, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 1.3619969843554676, derivative2_weight: -5.627672133620677, imbalance1_weights: [-1.2776465216319557, 0.3392523212525049, 1.5434923581961821, -0.9599514621291295,0.0], imbalance2_weights: [-0.0694206140631935, -0.4233320821592056, 0.8965599471012082, -0.3769300836604328, 0.0] }),
        down: Some(SignalParamsDir { threshold: 12.924718712398303, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.6160920532088923, derivative2_weight: 1.1051365224191168, imbalance1_weights: [0.09310995346508014, 0.005914599266595173, 0.015075183174709163, -0.21160337434994492,0.0], imbalance2_weights: [-0.06510006532673868, -0.07271140148367271, -0.008968957147040546, 0.028180256488763634, 0.0] })
    }
}

pub fn signal_trading_09_03() -> SignalParams {
    SignalParams {
        hold_ms: 250,
        up: Some(SignalParamsDir { threshold: 0.2547522778733997, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.6949038114054679, derivative2_weight: -0.9057303615846245, imbalance1_weights: [-0.10815244921679153, 0.12745537768125242, -0.072765326617233, -0.15667621777685273,0.0], imbalance2_weights: [0.27400577190429587, 0.20629281130053412, 0.26508092744809547, 0.19087896572348007, 0.0] }),
        down: Some(SignalParamsDir { threshold: 0.3538176827492965, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 1.248250215505707, derivative2_weight: 0.034994861974296965, imbalance1_weights: [0.02523119472401536, 0.02778552090110408, 0.06188950199514828, -0.0007433019868860787,0.0], imbalance2_weights: [0.018583702866224686, 0.06159849818772836, 0.005187903327439373, 7.727467309923685e-5, 0.0] })
    }
}

pub fn signal_huber_09_03() -> SignalParams {
    SignalParams {
        hold_ms: 1000,
        up: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.05124326526157465, derivative2_weight: -0.19611941487576973, imbalance1_weights: [-0.06253581560504573, 0.006939351480644682, 0.10987997903993134, -0.03269752875283044, -0.23285858029511453], imbalance2_weights: [-0.0038456095672108766, -0.05464831577343052, 0.06177953083378161, -0.07869529958228991, 0.10251479132464592] }),
        down: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.07229755635857968, derivative2_weight: 0.16600884508079192, imbalance1_weights: [0.032442815371301906, 0.0074899176730937565, 0.005448492115797206, -0.055317642278372, -0.0444462774148273], imbalance2_weights: [-0.04214605310253748, -0.022014509953197058, 0.0004923230976751271, -0.006873286371016392, -0.007872632266829117] })
    }
}

pub fn signal_huber_09_04() -> SignalParams {
    SignalParams {
        hold_ms: 1000,
        up: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.03752446308188132, derivative2_weight: -0.13340620473375148, imbalance1_weights: [-0.037600003644565505, -0.003126892959996035, 0.04618174829682374, -0.02955628253310262, -0.20247262189966858], imbalance2_weights: [0.03436751703867742, -0.06332656382957595, 0.03059486585307496, -0.039154366144395554, 0.013356634547231561] }),
        down: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.06398154182995208, derivative2_weight: 0.2055751905139963, imbalance1_weights: [0.017366922775181627, 0.003743563350690843, -0.009675251789126964, -0.00349574349458698, -0.0712659511429955], imbalance2_weights: [-0.014291682442279359, -0.02351418846219415, 0.00764053253693071, 0.0246153499281594, -0.09914070712857273] }),
    }
}

pub fn signal_huber_09_07() -> SignalParams {
    SignalParams {
        hold_ms: 1000,
        up: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.048408128876149134, derivative2_weight: -0.1689423808975731, imbalance1_weights: [-0.04245920649110392, -0.021611291692425316, 0.04855492208067832, -0.037268480646141425, -0.15538943718902745], imbalance2_weights: [0.046367943908634454, -0.06712417193397209, 0.012797486428465081, -0.026729165847502116, -0.010309443129275401] }),
        down: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.051621978511443484, derivative2_weight: 0.18139436776636744, imbalance1_weights: [0.011478052451157862, -0.002405933473071178, -0.001487197856530924, 0.010238596978191438, -0.07267383512291548], imbalance2_weights: [-0.016412370146882724, -0.00993269277684694, -0.022354722314475214, 0.03464565924773746, -0.05682647507824938] }),
    }
}

pub fn signal_huber_09_08() -> SignalParams {
    SignalParams {
        hold_ms: 1000,
        up: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.045471891643230775, derivative2_weight: -0.15595660860289298, imbalance1_weights: [-0.013611551590986325, -0.020486074459217567, 0.008784039271630344, 0.007415096449582987, 0.0166902174450521], imbalance2_weights: [0.013880029282235036, 0.019606786729914605, -0.06661747941657936, 0.05026987484758226, -0.025212617665203385] }),
        down: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.04080720941697665, derivative2_weight: 0.16916090035992198, imbalance1_weights: [0.006831938770685003, -0.006807744422032417, 0.0011129695797601272, 0.03384105040950336, -0.007785425168142405], imbalance2_weights: [-0.004049513024411125, -0.012068116988040925, -0.01116539440191004, 0.03880404595083105, -0.07129133860338088] })
    }
}

pub fn signal_huber_09_09() -> SignalParams {
    SignalParams {
        hold_ms: 1000,
        up: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.0581108733639847, derivative2_weight: -0.19024720104701323, imbalance1_weights: [-0.007047332627658196, -0.0069194278177951716, 0.0011545607615218046, 0.03699306740256864, 0.012365960125541477], imbalance2_weights: [-0.0014390777761287153, 0.04274675987378802, -0.07187359581231988, 0.03827845258588312, 0.03453788440879871] }),
        down: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.07894701464895088, derivative2_weight: 0.1937534287045527, imbalance1_weights: [0.000741361428454485, 0.00010959022467823962, 0.01948202810332367, 0.007810367383638151, 0.061665805893920206], imbalance2_weights: [-0.00399538698764593, -0.0015167844327072162, -0.031583004695106304, -0.007368055370681283, 0.05449377545478154] })
    }
}

pub fn signal_huber_09_10() -> SignalParams {
    SignalParams {
        hold_ms: 1000,
        up: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.058405422832820617, derivative2_weight: -0.20443549088300414, imbalance1_weights: [-0.006573787087795572, -0.006834921292407935, 0.002949456791672198, 0.024254524589924227, -0.06752948163103091], imbalance2_weights: [-0.0022684800798347883, 0.018262562232023688, -0.04135176973545457, 0.06831140352645991, -0.007233980760622014] }),
        down: Some(SignalParamsDir { threshold: 0.0, c_range: Range::from(-1.0 .. 1.0), a_range: Range::from(-1.0 .. 1.0), derivative1_weight: 0.06662065334176487, derivative2_weight: 0.19802859263659398, imbalance1_weights: [0.01676751934202555, 0.0009685164785355313, 0.02138390678790679, 0.03220681531074121, -0.011717230489880512], imbalance2_weights: [-0.0041418731045218355, 0.002701072439115374, 0.003683293111450568, -0.013886894459374133, 0.0055827973629622735] })
    }
}