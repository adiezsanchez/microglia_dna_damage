/*
OpenCL RandomForestClassifier
classifier_class_name = ObjectSegmenter
feature_specification = original gaussian_blur=1 gaussian_blur=2 gaussian_blur=3 gaussian_blur=4 gaussian_blur=5 difference_of_gaussian=1 difference_of_gaussian=2 difference_of_gaussian=3 difference_of_gaussian=4 difference_of_gaussian=5 laplace_box_of_gaussian_blur=1 laplace_box_of_gaussian_blur=2 laplace_box_of_gaussian_blur=3 laplace_box_of_gaussian_blur=4 laplace_box_of_gaussian_blur=5
num_ground_truth_dimensions = 2
num_classes = 2
num_features = 16
max_depth = 2
num_trees = 100
feature_importances = 0.0,0.012297010044596282,0.0,0.0,0.0,0.00024374536479401424,0.025085877551212037,0.05351704791935256,0.23552308736665764,0.1229289118082947,0.02356785969277793,0.01461906798008464,0.20396575075576262,0.2187995458597594,0.07054943644742366,0.018902659209284516
positive_class_identifier = 2
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_in4_TYPE in4, IMAGE_in5_TYPE in5, IMAGE_in6_TYPE in6, IMAGE_in7_TYPE in7, IMAGE_in8_TYPE in8, IMAGE_in9_TYPE in9, IMAGE_in10_TYPE in10, IMAGE_in11_TYPE in11, IMAGE_in12_TYPE in12, IMAGE_in13_TYPE in13, IMAGE_in14_TYPE in14, IMAGE_in15_TYPE in15, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float i4 = READ_IMAGE(in4, sampler, POS_in4_INSTANCE(x,y,z,0)).x;
 float i5 = READ_IMAGE(in5, sampler, POS_in5_INSTANCE(x,y,z,0)).x;
 float i6 = READ_IMAGE(in6, sampler, POS_in6_INSTANCE(x,y,z,0)).x;
 float i7 = READ_IMAGE(in7, sampler, POS_in7_INSTANCE(x,y,z,0)).x;
 float i8 = READ_IMAGE(in8, sampler, POS_in8_INSTANCE(x,y,z,0)).x;
 float i9 = READ_IMAGE(in9, sampler, POS_in9_INSTANCE(x,y,z,0)).x;
 float i10 = READ_IMAGE(in10, sampler, POS_in10_INSTANCE(x,y,z,0)).x;
 float i11 = READ_IMAGE(in11, sampler, POS_in11_INSTANCE(x,y,z,0)).x;
 float i12 = READ_IMAGE(in12, sampler, POS_in12_INSTANCE(x,y,z,0)).x;
 float i13 = READ_IMAGE(in13, sampler, POS_in13_INSTANCE(x,y,z,0)).x;
 float i14 = READ_IMAGE(in14, sampler, POS_in14_INSTANCE(x,y,z,0)).x;
 float i15 = READ_IMAGE(in15, sampler, POS_in15_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i12<6.123720169067383){
 if(i14<2.0044965744018555){
  s0+=0.9970816475426145;
  s1+=0.0029183524573854216;
 } else {
  s0+=0.3364485981308411;
  s1+=0.6635514018691588;
 }
} else {
 if(i12<8.16076946258545){
  s0+=0.2669491525423729;
  s1+=0.7330508474576272;
 } else {
  s0+=0.008966493629070316;
  s1+=0.9910335063709297;
 }
}
if(i13<3.1350831985473633){
 if(i7<1.7919597625732422){
  s0+=0.9947096944848565;
  s1+=0.005290305515143499;
 } else {
  s0+=0.14130434782608695;
  s1+=0.8586956521739131;
 }
} else {
 if(i14<3.112884521484375){
  s0+=0.2433392539964476;
  s1+=0.7566607460035524;
 } else {
  s0+=0.006732263076126359;
  s1+=0.9932677369238736;
 }
}
if(i13<2.975107192993164){
 if(i11<20.709352493286133){
  s0+=0.9949114459423738;
  s1+=0.005088554057626223;
 } else {
  s0+=0.26136363636363635;
  s1+=0.7386363636363636;
 }
} else {
 if(i13<4.495262145996094){
  s0+=0.3626666666666667;
  s1+=0.6373333333333333;
 } else {
  s0+=0.019971469329529243;
  s1+=0.9800285306704708;
 }
}
if(i13<3.0059032440185547){
 if(i11<18.769262313842773){
  s0+=0.9944734926202925;
  s1+=0.005526507379707459;
 } else {
  s0+=0.36363636363636365;
  s1+=0.6363636363636364;
 }
} else {
 if(i5<41.90266036987305){
  s0+=0.027160493827160494;
  s1+=0.9728395061728395;
 } else {
  s0+=0.26024590163934425;
  s1+=0.7397540983606558;
 }
}
if(i15<1.3846778869628906){
 if(i13<2.891360282897949){
  s0+=0.9949740380774863;
  s1+=0.005025961922513647;
 } else {
  s0+=0.19938650306748465;
  s1+=0.8006134969325154;
 }
} else {
 if(i9<2.774984359741211){
  s0+=0.6803519061583577;
  s1+=0.3196480938416422;
 } else {
  s0+=0.018404907975460124;
  s1+=0.9815950920245399;
 }
}
if(i1<41.445255279541016){
 if(i9<2.0805130004882812){
  s0+=0.9912322274881517;
  s1+=0.008767772511848342;
 } else {
  s0+=0.09210526315789473;
  s1+=0.9078947368421053;
 }
} else {
 if(i8<2.523649215698242){
  s0+=0.912961210974456;
  s1+=0.08703878902554399;
 } else {
  s0+=0.021945137157107233;
  s1+=0.9780548628428928;
 }
}
if(i12<5.581586837768555){
 if(i10<3.0822811126708984){
  s0+=0.99557901682613;
  s1+=0.00442098317387001;
 } else {
  s0+=0.21212121212121213;
  s1+=0.7878787878787878;
 }
} else {
 if(i8<3.232694625854492){
  s0+=0.26476190476190475;
  s1+=0.7352380952380952;
 } else {
  s0+=0.00273224043715847;
  s1+=0.9972677595628415;
 }
}
if(i8<1.9265151023864746){
 if(i11<20.819961547851562){
  s0+=0.9946170866219741;
  s1+=0.005382913378025825;
 } else {
  s0+=0.22666666666666666;
  s1+=0.7733333333333333;
 }
} else {
 if(i8<2.6103477478027344){
  s0+=0.38095238095238093;
  s1+=0.6190476190476191;
 } else {
  s0+=0.019990913221263062;
  s1+=0.980009086778737;
 }
}
if(i8<1.9240245819091797){
 if(i6<1.5370635986328125){
  s0+=0.9953714417958806;
  s1+=0.004628558204119417;
 } else {
  s0+=0.3135593220338983;
  s1+=0.6864406779661016;
 }
} else {
 if(i12<8.166685104370117){
  s0+=0.3060747663551402;
  s1+=0.6939252336448598;
 } else {
  s0+=0.006872852233676976;
  s1+=0.993127147766323;
 }
}
if(i13<2.9759445190429688){
 if(i6<1.5637493133544922){
  s0+=0.9943155529116267;
  s1+=0.005684447088373323;
 } else {
  s0+=0.3684210526315789;
  s1+=0.631578947368421;
 }
} else {
 if(i8<2.624879837036133){
  s0+=0.3943217665615142;
  s1+=0.6056782334384858;
 } else {
  s0+=0.013425925925925926;
  s1+=0.986574074074074;
 }
}
if(i13<3.1389455795288086){
 if(i8<1.3400659561157227){
  s0+=0.996990167881747;
  s1+=0.0030098321182529596;
 } else {
  s0+=0.7422303473491774;
  s1+=0.2577696526508227;
 }
} else {
 if(i12<8.574020385742188){
  s0+=0.2888402625820569;
  s1+=0.7111597374179431;
 } else {
  s0+=0.008835758835758836;
  s1+=0.9911642411642412;
 }
}
if(i14<1.9998903274536133){
 if(i11<22.633460998535156){
  s0+=0.9951011221078415;
  s1+=0.004898877892158485;
 } else {
  s0+=0.07518796992481203;
  s1+=0.924812030075188;
 }
} else {
 if(i9<2.6485061645507812){
  s0+=0.452991452991453;
  s1+=0.5470085470085471;
 } else {
  s0+=0.02975133214920071;
  s1+=0.9702486678507993;
 }
}
if(i14<1.9998903274536133){
 if(i11<22.775968551635742){
  s0+=0.9954432887568103;
  s1+=0.004556711243189698;
 } else {
  s0+=0.08496732026143791;
  s1+=0.9150326797385621;
 }
} else {
 if(i9<2.773956298828125){
  s0+=0.42024539877300615;
  s1+=0.5797546012269938;
 } else {
  s0+=0.022265246853823813;
  s1+=0.9777347531461762;
 }
}
if(i8<1.718675136566162){
 if(i13<1.6701288223266602){
  s0+=0.998869746891804;
  s1+=0.0011302531081960475;
 } else {
  s0+=0.8765323992994746;
  s1+=0.1234676007005254;
 }
} else {
 if(i9<2.987668037414551){
  s0+=0.30812854442344045;
  s1+=0.6918714555765595;
 } else {
  s0+=0.009683995922528032;
  s1+=0.990316004077472;
 }
}
if(i13<3.144500732421875){
 if(i11<20.819961547851562){
  s0+=0.9948276998089214;
  s1+=0.0051723001910786055;
 } else {
  s0+=0.23333333333333334;
  s1+=0.7666666666666667;
 }
} else {
 if(i7<2.2291812896728516){
  s0+=0.29411764705882354;
  s1+=0.7058823529411765;
 } else {
  s0+=0.011728709841917389;
  s1+=0.9882712901580826;
 }
}
if(i12<5.606952667236328){
 if(i13<2.9759445190429688){
  s0+=0.9957977632188472;
  s1+=0.004202236781152803;
 } else {
  s0+=0.33505154639175255;
  s1+=0.6649484536082474;
 }
} else {
 if(i8<3.0048885345458984){
  s0+=0.30823529411764705;
  s1+=0.691764705882353;
 } else {
  s0+=0.004022121669180492;
  s1+=0.9959778783308195;
 }
}
if(i13<2.8799667358398438){
 if(i9<2.2372922897338867){
  s0+=0.9947716743878227;
  s1+=0.005228325612177366;
 } else {
  s0+=0.359375;
  s1+=0.640625;
 }
} else {
 if(i14<2.7962865829467773){
  s0+=0.2796934865900383;
  s1+=0.7203065134099617;
 } else {
  s0+=0.01432806324110672;
  s1+=0.9856719367588933;
 }
}
if(i9<2.183903694152832){
 if(i13<3.158782958984375){
  s0+=0.9933300313686644;
  s1+=0.006669968631335644;
 } else {
  s0+=0.11428571428571428;
  s1+=0.8857142857142857;
 }
} else {
 if(i15<1.9913463592529297){
  s0+=0.2674616695059625;
  s1+=0.7325383304940375;
 } else {
  s0+=0.009713977334052886;
  s1+=0.9902860226659471;
 }
}
if(i12<5.583160400390625){
 if(i13<2.8306970596313477){
  s0+=0.9955191184280403;
  s1+=0.004480881571959639;
 } else {
  s0+=0.3868312757201646;
  s1+=0.6131687242798354;
 }
} else {
 if(i14<2.9645910263061523){
  s0+=0.22882882882882882;
  s1+=0.7711711711711712;
 } else {
  s0+=0.007352941176470588;
  s1+=0.9926470588235294;
 }
}
if(i9<2.1009159088134766){
 if(i12<6.389441967010498){
  s0+=0.9970838718229115;
  s1+=0.002916128177088511;
 } else {
  s0+=0.15463917525773196;
  s1+=0.845360824742268;
 }
} else {
 if(i13<5.114975929260254){
  s0+=0.3558648111332008;
  s1+=0.6441351888667992;
 } else {
  s0+=0.004601226993865031;
  s1+=0.995398773006135;
 }
}
if(i8<1.9176502227783203){
 if(i14<1.9407835006713867){
  s0+=0.9945747461047338;
  s1+=0.005425253895266135;
 } else {
  s0+=0.5341614906832298;
  s1+=0.4658385093167702;
 }
} else {
 if(i8<2.624879837036133){
  s0+=0.3688212927756654;
  s1+=0.6311787072243346;
 } else {
  s0+=0.01929260450160772;
  s1+=0.9807073954983923;
 }
}
if(i7<1.5364952087402344){
 if(i15<1.4768142700195312){
  s0+=0.9978722696898168;
  s1+=0.0021277303101831842;
 } else {
  s0+=0.42021276595744683;
  s1+=0.5797872340425532;
 }
} else {
 if(i12<7.157524108886719){
  s0+=0.459915611814346;
  s1+=0.540084388185654;
 } else {
  s0+=0.021515434985968196;
  s1+=0.9784845650140318;
 }
}
if(i8<1.7458019256591797){
 if(i6<1.6571769714355469){
  s0+=0.9950842928309854;
  s1+=0.004915707169014549;
 } else {
  s0+=0.37681159420289856;
  s1+=0.6231884057971014;
 }
} else {
 if(i12<8.162986755371094){
  s0+=0.35526315789473684;
  s1+=0.6447368421052632;
 } else {
  s0+=0.01354062186559679;
  s1+=0.9864593781344032;
 }
}
if(i12<6.051443099975586){
 if(i8<1.7778587341308594){
  s0+=0.9964338781575037;
  s1+=0.003566121842496285;
 } else {
  s0+=0.3159851301115242;
  s1+=0.6840148698884758;
 }
} else {
 if(i13<4.320093154907227){
  s0+=0.2897959183673469;
  s1+=0.710204081632653;
 } else {
  s0+=0.008370260955194485;
  s1+=0.9916297390448056;
 }
}
if(i13<2.975107192993164){
 if(i6<1.5370635986328125){
  s0+=0.995405262461986;
  s1+=0.004594737538014016;
 } else {
  s0+=0.40860215053763443;
  s1+=0.5913978494623656;
 }
} else {
 if(i12<8.574020385742188){
  s0+=0.2693069306930693;
  s1+=0.7306930693069307;
 } else {
  s0+=0.007575757575757576;
  s1+=0.9924242424242424;
 }
}
if(i8<1.7351818084716797){
 if(i6<1.5196805000305176){
  s0+=0.9954587642535137;
  s1+=0.0045412357464863435;
 } else {
  s0+=0.41;
  s1+=0.59;
 }
} else {
 if(i11<9.847633361816406){
  s0+=0.21200750469043153;
  s1+=0.7879924953095685;
 } else {
  s0+=0.027106949236076885;
  s1+=0.9728930507639231;
 }
}
if(i12<6.356376647949219){
 if(i8<1.7303495407104492){
  s0+=0.9958374628344896;
  s1+=0.004162537165510407;
 } else {
  s0+=0.4163568773234201;
  s1+=0.5836431226765799;
 }
} else {
 if(i9<3.091510772705078){
  s0+=0.14657210401891252;
  s1+=0.8534278959810875;
 } else {
  s0+=0.0026766595289079227;
  s1+=0.9973233404710921;
 }
}
if(i9<2.189493179321289){
 if(i6<1.679433822631836){
  s0+=0.9949747082355275;
  s1+=0.00502529176447251;
 } else {
  s0+=0.1267605633802817;
  s1+=0.8732394366197183;
 }
} else {
 if(i8<2.6914405822753906){
  s0+=0.41304347826086957;
  s1+=0.5869565217391305;
 } else {
  s0+=0.015101462954223691;
  s1+=0.9848985370457763;
 }
}
if(i14<2.0044965744018555){
 if(i12<6.680271625518799){
  s0+=0.9964569536423841;
  s1+=0.003543046357615894;
 } else {
  s0+=0.11734693877551021;
  s1+=0.8826530612244898;
 }
} else {
 if(i13<4.481876373291016){
  s0+=0.3633720930232558;
  s1+=0.6366279069767442;
 } else {
  s0+=0.01291866028708134;
  s1+=0.9870813397129187;
 }
}
if(i13<2.8896360397338867){
 if(i6<1.5463223457336426){
  s0+=0.9960352859550005;
  s1+=0.003964714044999504;
 } else {
  s0+=0.4117647058823529;
  s1+=0.5882352941176471;
 }
} else {
 if(i10<3.7001171112060547){
  s0+=0.18696186961869618;
  s1+=0.8130381303813038;
 } else {
  s0+=0.0018018018018018018;
  s1+=0.9981981981981982;
 }
}
if(i10<2.324148654937744){
 if(i8<1.9258809089660645){
  s0+=0.9959168769087771;
  s1+=0.004083123091222945;
 } else {
  s0+=0.12162162162162163;
  s1+=0.8783783783783784;
 }
} else {
 if(i9<2.7023868560791016){
  s0+=0.7114093959731543;
  s1+=0.28859060402684567;
 } else {
  s0+=0.02793560606060606;
  s1+=0.9720643939393939;
 }
}
if(i13<2.975107192993164){
 if(i12<5.925793170928955){
  s0+=0.9953116745905969;
  s1+=0.004688325409403064;
 } else {
  s0+=0.27184466019417475;
  s1+=0.7281553398058253;
 }
} else {
 if(i14<2.7959375381469727){
  s0+=0.2943820224719101;
  s1+=0.7056179775280899;
 } else {
  s0+=0.0160481444332999;
  s1+=0.9839518555667001;
 }
}
if(i7<1.5456438064575195){
 if(i9<2.4930171966552734){
  s0+=0.9964972572863657;
  s1+=0.0035027427136342607;
 } else {
  s0+=0.225;
  s1+=0.775;
 }
} else {
 if(i12<8.573200225830078){
  s0+=0.2992125984251969;
  s1+=0.7007874015748031;
 } else {
  s0+=0.00754906894816306;
  s1+=0.9924509310518369;
 }
}
if(i12<5.581586837768555){
 if(i15<1.473684310913086){
  s0+=0.9971969166082691;
  s1+=0.002803083391730904;
 } else {
  s0+=0.4492753623188406;
  s1+=0.5507246376811594;
 }
} else {
 if(i9<3.128615379333496){
  s0+=0.24236641221374045;
  s1+=0.7576335877862596;
 } else {
  s0+=0.006519558676028084;
  s1+=0.993480441323972;
 }
}
if(i12<6.37837028503418){
 if(i14<2.1431522369384766){
  s0+=0.9962765256359563;
  s1+=0.003723474364043759;
 } else {
  s0+=0.2909090909090909;
  s1+=0.7090909090909091;
 }
} else {
 if(i7<2.3330440521240234){
  s0+=0.24497991967871485;
  s1+=0.7550200803212851;
 } else {
  s0+=0.0056179775280898875;
  s1+=0.9943820224719101;
 }
}
if(i9<2.2372264862060547){
 if(i11<22.775968551635742){
  s0+=0.994813346547737;
  s1+=0.005186653452262967;
 } else {
  s0+=0.07792207792207792;
  s1+=0.922077922077922;
 }
} else {
 if(i13<4.3478546142578125){
  s0+=0.41403508771929826;
  s1+=0.5859649122807018;
 } else {
  s0+=0.014615747289014616;
  s1+=0.9853842527109854;
 }
}
if(i7<1.629617691040039){
 if(i10<3.0000381469726562){
  s0+=0.9958948553267563;
  s1+=0.004105144673243726;
 } else {
  s0+=0.31004366812227074;
  s1+=0.6899563318777293;
 }
} else {
 if(i13<4.364521026611328){
  s0+=0.31756756756756754;
  s1+=0.6824324324324325;
 } else {
  s0+=0.008099094807050978;
  s1+=0.991900905192949;
 }
}
if(i12<5.606952667236328){
 if(i8<1.7498435974121094){
  s0+=0.994879249397073;
  s1+=0.005120750602927087;
 } else {
  s0+=0.3093220338983051;
  s1+=0.690677966101695;
 }
} else {
 if(i9<3.4293880462646484){
  s0+=0.2179261862917399;
  s1+=0.7820738137082601;
 } else {
  s0+=0.007972665148063782;
  s1+=0.9920273348519362;
 }
}
if(i14<2.0288877487182617){
 if(i12<6.123720169067383){
  s0+=0.9968866955916934;
  s1+=0.003113304408306561;
 } else {
  s0+=0.13777777777777778;
  s1+=0.8622222222222222;
 }
} else {
 if(i13<4.793901443481445){
  s0+=0.3415233415233415;
  s1+=0.6584766584766585;
 } else {
  s0+=0.008977556109725686;
  s1+=0.9910224438902743;
 }
}
if(i8<1.8026103973388672){
 if(i12<5.776680946350098){
  s0+=0.9959006909319317;
  s1+=0.004099309068068366;
 } else {
  s0+=0.36065573770491804;
  s1+=0.639344262295082;
 }
} else {
 if(i7<2.3458080291748047){
  s0+=0.30401529636711283;
  s1+=0.6959847036328872;
 } else {
  s0+=0.006714876033057851;
  s1+=0.9932851239669421;
 }
}
if(i8<1.7778587341308594){
 if(i15<1.473341941833496){
  s0+=0.9954838281198114;
  s1+=0.004516171880188617;
 } else {
  s0+=0.6482412060301508;
  s1+=0.35175879396984927;
 }
} else {
 if(i8<2.644519805908203){
  s0+=0.4036144578313253;
  s1+=0.5963855421686747;
 } else {
  s0+=0.013729977116704805;
  s1+=0.9862700228832952;
 }
}
if(i13<3.15584659576416){
 if(i7<1.9591679573059082){
  s0+=0.9941046668642756;
  s1+=0.005895333135724401;
 } else {
  s0+=0.12698412698412698;
  s1+=0.873015873015873;
 }
} else {
 if(i8<3.0048885345458984){
  s0+=0.27927927927927926;
  s1+=0.7207207207207207;
 } else {
  s0+=0.00510204081632653;
  s1+=0.9948979591836735;
 }
}
if(i12<6.111056327819824){
 if(i9<2.2332420349121094){
  s0+=0.9970589207230428;
  s1+=0.0029410792769571396;
 } else {
  s0+=0.3006535947712418;
  s1+=0.6993464052287581;
 }
} else {
 if(i14<2.7739295959472656){
  s0+=0.14987714987714987;
  s1+=0.8501228501228502;
 } else {
  s0+=0.007004310344827586;
  s1+=0.9929956896551724;
 }
}
if(i12<5.781133651733398){
 if(i8<1.7526321411132812){
  s0+=0.9958066433335535;
  s1+=0.0041933566664465426;
 } else {
  s0+=0.3438735177865613;
  s1+=0.6561264822134387;
 }
} else {
 if(i11<22.781770706176758){
  s0+=0.1291005291005291;
  s1+=0.870899470899471;
 } else {
  s0+=0.009658246656760773;
  s1+=0.9903417533432393;
 }
}
if(i12<5.885358810424805){
 if(i9<2.373532295227051){
  s0+=0.9964241962718935;
  s1+=0.0035758037281064794;
 } else {
  s0+=0.2620967741935484;
  s1+=0.7379032258064516;
 }
} else {
 if(i8<2.8101558685302734){
  s0+=0.24354243542435425;
  s1+=0.7564575645756457;
 } else {
  s0+=0.011385199240986717;
  s1+=0.9886148007590133;
 }
}
if(i8<1.7743005752563477){
 if(i14<2.3057546615600586){
  s0+=0.9950166661166299;
  s1+=0.0049833338833701855;
 } else {
  s0+=0.05405405405405406;
  s1+=0.9459459459459459;
 }
} else {
 if(i12<8.573200225830078){
  s0+=0.318359375;
  s1+=0.681640625;
 } else {
  s0+=0.0035353535353535356;
  s1+=0.9964646464646465;
 }
}
if(i9<2.073519706726074){
 if(i8<1.9258737564086914){
  s0+=0.9950923882612562;
  s1+=0.004907611738743783;
 } else {
  s0+=0.10714285714285714;
  s1+=0.8928571428571429;
 }
} else {
 if(i11<8.937433242797852){
  s0+=0.25;
  s1+=0.75;
 } else {
  s0+=0.026217228464419477;
  s1+=0.9737827715355806;
 }
}
if(i13<2.8896360397338867){
 if(i9<2.2545766830444336){
  s0+=0.9948393926362102;
  s1+=0.005160607363789738;
 } else {
  s0+=0.39344262295081966;
  s1+=0.6065573770491803;
 }
} else {
 if(i5<43.615150451660156){
  s0+=0.022180273714016045;
  s1+=0.9778197262859839;
 } else {
  s0+=0.30641330166270786;
  s1+=0.6935866983372921;
 }
}
if(i12<6.471688270568848){
 if(i15<1.4350719451904297){
  s0+=0.9969722176010647;
  s1+=0.003027782398935285;
 } else {
  s0+=0.48008849557522126;
  s1+=0.5199115044247787;
 }
} else {
 if(i13<4.255620956420898){
  s0+=0.27751196172248804;
  s1+=0.722488038277512;
 } else {
  s0+=0.01229895931882687;
  s1+=0.9877010406811731;
 }
}
if(i15<1.3895130157470703){
 if(i11<18.242610931396484){
  s0+=0.9958153437396214;
  s1+=0.004184656260378612;
 } else {
  s0+=0.18670886075949367;
  s1+=0.8132911392405063;
 }
} else {
 if(i7<1.2252388000488281){
  s0+=0.5815384615384616;
  s1+=0.41846153846153844;
 } else {
  s0+=0.03607503607503607;
  s1+=0.963924963924964;
 }
}
if(i8<1.7709484100341797){
 if(i9<2.083646774291992){
  s0+=0.9953651592398861;
  s1+=0.0046348407601138845;
 } else {
  s0+=0.4728682170542636;
  s1+=0.5271317829457365;
 }
} else {
 if(i12<8.18610954284668){
  s0+=0.34451901565995524;
  s1+=0.6554809843400448;
 } else {
  s0+=0.01025390625;
  s1+=0.98974609375;
 }
}
if(i8<1.718675136566162){
 if(i12<6.123720169067383){
  s0+=0.9952321038341831;
  s1+=0.004767896165816833;
 } else {
  s0+=0.21311475409836064;
  s1+=0.7868852459016393;
 }
} else {
 if(i7<2.603959083557129){
  s0+=0.26300148588410105;
  s1+=0.736998514115899;
 } else {
  s0+=0.0021119324181626186;
  s1+=0.9978880675818373;
 }
}
if(i1<40.817169189453125){
 if(i8<1.616607666015625){
  s0+=0.9957559501578787;
  s1+=0.0042440498421213455;
 } else {
  s0+=0.1605351170568562;
  s1+=0.8394648829431438;
 }
} else {
 if(i9<2.639634132385254){
  s0+=0.9187913125590179;
  s1+=0.08120868744098206;
 } else {
  s0+=0.030212976721149084;
  s1+=0.9697870232788509;
 }
}
if(i8<1.850614070892334){
 if(i14<2.0044965744018555){
  s0+=0.9948359760336324;
  s1+=0.005164023966367639;
 } else {
  s0+=0.4725274725274725;
  s1+=0.5274725274725275;
 }
} else {
 if(i13<4.766087532043457){
  s0+=0.31381733021077285;
  s1+=0.6861826697892272;
 } else {
  s0+=0.008559201141226819;
  s1+=0.9914407988587732;
 }
}
if(i6<1.201345443725586){
 if(i7<1.5371570587158203){
  s0+=0.992057214422253;
  s1+=0.007942785577747018;
 } else {
  s0+=0.13114754098360656;
  s1+=0.8688524590163934;
 }
} else {
 if(i6<1.4609394073486328){
  s0+=0.33774834437086093;
  s1+=0.6622516556291391;
 } else {
  s0+=0.041584158415841586;
  s1+=0.9584158415841584;
 }
}
if(i14<2.0214881896972656){
 if(i7<1.8077430725097656){
  s0+=0.9959990741659227;
  s1+=0.004000925834077307;
 } else {
  s0+=0.10204081632653061;
  s1+=0.8979591836734694;
 }
} else {
 if(i14<2.7959375381469727){
  s0+=0.35;
  s1+=0.65;
 } else {
  s0+=0.0174042764793635;
  s1+=0.9825957235206365;
 }
}
if(i8<1.7302069664001465){
 if(i7<1.782090663909912){
  s0+=0.9948778956412544;
  s1+=0.00512210435874558;
 } else {
  s0+=0.07142857142857142;
  s1+=0.9285714285714286;
 }
} else {
 if(i13<4.147947311401367){
  s0+=0.3945945945945946;
  s1+=0.6054054054054054;
 } else {
  s0+=0.02053196453569762;
  s1+=0.9794680354643024;
 }
}
if(i13<3.1389455795288086){
 if(i12<5.834283828735352){
  s0+=0.9953294246248634;
  s1+=0.004670575375136639;
 } else {
  s0+=0.3208955223880597;
  s1+=0.6791044776119403;
 }
} else {
 if(i8<3.1558895111083984){
  s0+=0.26396917148362237;
  s1+=0.7360308285163777;
 } else {
  s0+=0.001006036217303823;
  s1+=0.9989939637826962;
 }
}
if(i14<1.9998903274536133){
 if(i11<21.941892623901367){
  s0+=0.9955950054648428;
  s1+=0.004404994535157155;
 } else {
  s0+=0.10493827160493827;
  s1+=0.8950617283950617;
 }
} else {
 if(i14<2.790151596069336){
  s0+=0.3813131313131313;
  s1+=0.6186868686868687;
 } else {
  s0+=0.012506012506012507;
  s1+=0.9874939874939875;
 }
}
if(i9<2.0841712951660156){
 if(i12<6.947694778442383){
  s0+=0.996449900464499;
  s1+=0.0035500995355009952;
 } else {
  s0+=0.10967741935483871;
  s1+=0.8903225806451613;
 }
} else {
 if(i14<2.5794429779052734){
  s0+=0.4092219020172911;
  s1+=0.590778097982709;
 } else {
  s0+=0.019195612431444242;
  s1+=0.9808043875685558;
 }
}
if(i13<3.0786046981811523){
 if(i6<1.7665824890136719){
  s0+=0.9945130766197924;
  s1+=0.005486923380207649;
 } else {
  s0+=0.28125;
  s1+=0.71875;
 }
} else {
 if(i12<8.772212982177734){
  s0+=0.2857142857142857;
  s1+=0.7142857142857143;
 } else {
  s0+=0.0053533190578158455;
  s1+=0.9946466809421841;
 }
}
if(i13<3.1389455795288086){
 if(i8<1.3423221111297607){
  s0+=0.996854188280178;
  s1+=0.0031458117198219604;
 } else {
  s0+=0.7549909255898367;
  s1+=0.24500907441016334;
 }
} else {
 if(i13<4.329750061035156){
  s0+=0.3146853146853147;
  s1+=0.6853146853146853;
 } else {
  s0+=0.010416666666666666;
  s1+=0.9895833333333334;
 }
}
if(i12<6.123720169067383){
 if(i14<2.2957029342651367){
  s0+=0.9963884563135748;
  s1+=0.0036115436864252344;
 } else {
  s0+=0.23921568627450981;
  s1+=0.7607843137254902;
 }
} else {
 if(i8<3.1568307876586914){
  s0+=0.21875;
  s1+=0.78125;
 } else {
  s0+=0.003980099502487562;
  s1+=0.9960199004975124;
 }
}
if(i13<3.342656135559082){
 if(i11<18.989471435546875){
  s0+=0.9946576968737634;
  s1+=0.005342303126236644;
 } else {
  s0+=0.35;
  s1+=0.65;
 }
} else {
 if(i12<8.731382369995117){
  s0+=0.25968109339407747;
  s1+=0.7403189066059226;
 } else {
  s0+=0.004622496147919877;
  s1+=0.9953775038520801;
 }
}
if(i12<6.111056327819824){
 if(i9<2.2648725509643555){
  s0+=0.9961668098605512;
  s1+=0.003833190139448814;
 } else {
  s0+=0.2733118971061093;
  s1+=0.7266881028938906;
 }
} else {
 if(i15<2.2269210815429688){
  s0+=0.11891117478510028;
  s1+=0.8810888252148997;
 } else {
  s1+=1.0;
 }
}
if(i6<1.1181354522705078){
 if(i8<1.7458019256591797){
  s0+=0.9956246478172959;
  s1+=0.0043753521827041;
 } else {
  s0+=0.17669172932330826;
  s1+=0.8233082706766918;
 }
} else {
 if(i7<2.0280542373657227){
  s0+=0.7591836734693878;
  s1+=0.24081632653061225;
 } else {
  s0+=0.014215080346106305;
  s1+=0.9857849196538937;
 }
}
if(i8<1.8026103973388672){
 if(i12<5.9116997718811035){
  s0+=0.9954987754021315;
  s1+=0.0045012245978685375;
 } else {
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 }
} else {
 if(i7<2.3458080291748047){
  s0+=0.31499051233396586;
  s1+=0.6850094876660342;
 } else {
  s0+=0.009504752376188095;
  s1+=0.9904952476238119;
 }
}
if(i9<2.0841712951660156){
 if(i13<2.8863344192504883){
  s0+=0.9951263178834295;
  s1+=0.00487368211657052;
 } else {
  s0+=0.2236842105263158;
  s1+=0.7763157894736842;
 }
} else {
 if(i9<2.773956298828125){
  s0+=0.4049586776859504;
  s1+=0.5950413223140496;
 } else {
  s0+=0.021365536460752437;
  s1+=0.9786344635392475;
 }
}
if(i12<5.741189002990723){
 if(i14<2.1017942428588867){
  s0+=0.9968907118285261;
  s1+=0.003109288171473935;
 } else {
  s0+=0.3102189781021898;
  s1+=0.6897810218978102;
 }
} else {
 if(i12<8.371309280395508){
  s0+=0.33116883116883117;
  s1+=0.6688311688311688;
 } else {
  s0+=0.00992063492063492;
  s1+=0.9900793650793651;
 }
}
if(i13<2.975107192993164){
 if(i11<19.004188537597656){
  s0+=0.9951983575071197;
  s1+=0.004801642492880323;
 } else {
  s0+=0.3626373626373626;
  s1+=0.6373626373626373;
 }
} else {
 if(i13<3.7744078636169434){
  s0+=0.4581497797356828;
  s1+=0.5418502202643172;
 } else {
  s0+=0.020743301642178046;
  s1+=0.9792566983578219;
 }
}
if(i8<1.8026103973388672){
 if(i11<18.769262313842773){
  s0+=0.9955562924888078;
  s1+=0.0044437075111921735;
 } else {
  s0+=0.35135135135135137;
  s1+=0.6486486486486487;
 }
} else {
 if(i7<2.0280542373657227){
  s0+=0.3074792243767313;
  s1+=0.6925207756232687;
 } else {
  s0+=0.01407172038129823;
  s1+=0.9859282796187018;
 }
}
if(i8<1.9495763778686523){
 if(i11<20.758544921875){
  s0+=0.9944393261384575;
  s1+=0.005560673861542511;
 } else {
  s0+=0.2727272727272727;
  s1+=0.7272727272727273;
 }
} else {
 if(i13<4.746866226196289){
  s0+=0.31521739130434784;
  s1+=0.6847826086956522;
 } else {
  s0+=0.008072653884964682;
  s1+=0.9919273461150353;
 }
}
if(i12<6.470185279846191){
 if(i10<2.47921085357666){
  s0+=0.9967119465940416;
  s1+=0.003288053405958351;
 } else {
  s0+=0.4676724137931034;
  s1+=0.5323275862068966;
 }
} else {
 if(i7<2.6075940132141113){
  s0+=0.1956043956043956;
  s1+=0.8043956043956044;
 } else {
  s1+=1.0;
 }
}
if(i8<1.8219661712646484){
 if(i11<20.819961547851562){
  s0+=0.9956754258550112;
  s1+=0.004324574144988776;
 } else {
  s0+=0.35064935064935066;
  s1+=0.6493506493506493;
 }
} else {
 if(i8<2.9257469177246094){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s0+=0.01033464566929134;
  s1+=0.9896653543307087;
 }
}
if(i9<2.2370786666870117){
 if(i12<6.12576961517334){
  s0+=0.9968804964656689;
  s1+=0.003119503534331132;
 } else {
  s0+=0.20647773279352227;
  s1+=0.7935222672064778;
 }
} else {
 if(i8<3.100008487701416){
  s0+=0.34210526315789475;
  s1+=0.6578947368421053;
 } else {
  s0+=0.006018054162487462;
  s1+=0.9939819458375125;
 }
}
if(i11<13.803959846496582){
 if(i10<3.0028724670410156){
  s0+=0.9941634886420162;
  s1+=0.00583651135798375;
 } else {
  s0+=0.13660245183887915;
  s1+=0.8633975481611208;
 }
} else {
 if(i14<1.7462306022644043){
  s0+=0.5625;
  s1+=0.4375;
 } else {
  s0+=0.019796380090497737;
  s1+=0.9802036199095022;
 }
}
if(i13<2.9434657096862793){
 if(i13<2.499673843383789){
  s0+=0.9955929619934392;
  s1+=0.004407038006560854;
 } else {
  s0+=0.6901960784313725;
  s1+=0.30980392156862746;
 }
} else {
 if(i13<4.355199813842773){
  s0+=0.3355048859934853;
  s1+=0.6644951140065146;
 } else {
  s0+=0.01675442795595979;
  s1+=0.9832455720440402;
 }
}
if(i8<1.7778587341308594){
 if(i9<2.0805130004882812){
  s0+=0.995266154661017;
  s1+=0.004733845338983051;
 } else {
  s0+=0.525;
  s1+=0.475;
 }
} else {
 if(i8<2.691892623901367){
  s0+=0.42;
  s1+=0.58;
 } else {
  s0+=0.013475836431226766;
  s1+=0.9865241635687733;
 }
}
if(i12<5.614900588989258){
 if(i9<2.614699363708496){
  s0+=0.9963726298433636;
  s1+=0.0036273701566364386;
 } else {
  s0+=0.16374269005847952;
  s1+=0.8362573099415205;
 }
} else {
 if(i14<2.7718048095703125){
  s0+=0.24081632653061225;
  s1+=0.7591836734693878;
 } else {
  s0+=0.010303687635574838;
  s1+=0.9896963123644251;
 }
}
if(i9<2.106945037841797){
 if(i6<1.5637493133544922){
  s0+=0.9946880913648285;
  s1+=0.005311908635171475;
 } else {
  s0+=0.20218579234972678;
  s1+=0.7978142076502732;
 }
} else {
 if(i14<2.580984115600586){
  s0+=0.4431818181818182;
  s1+=0.5568181818181818;
 } else {
  s0+=0.027598896044158234;
  s1+=0.9724011039558418;
 }
}
if(i12<6.393856048583984){
 if(i10<3.046576499938965){
  s0+=0.9950253673321473;
  s1+=0.004974632667852672;
 } else {
  s0+=0.29218106995884774;
  s1+=0.7078189300411523;
 }
} else {
 if(i12<8.201641082763672){
  s0+=0.23605150214592274;
  s1+=0.7639484978540773;
 } else {
  s0+=0.01;
  s1+=0.99;
 }
}
if(i7<1.597646713256836){
 if(i9<2.1777524948120117){
  s0+=0.9972452704945237;
  s1+=0.0027547295054762694;
 } else {
  s0+=0.36227544910179643;
  s1+=0.6377245508982036;
 }
} else {
 if(i12<9.264840126037598){
  s0+=0.2597938144329897;
  s1+=0.7402061855670103;
 } else {
  s0+=0.003189792663476874;
  s1+=0.9968102073365231;
 }
}
if(i12<6.434521675109863){
 if(i14<2.2946529388427734){
  s0+=0.9961924312154422;
  s1+=0.0038075687845578253;
 } else {
  s0+=0.23954372623574144;
  s1+=0.7604562737642585;
 }
} else {
 if(i14<2.864267349243164){
  s0+=0.15955056179775282;
  s1+=0.8404494382022472;
 } else {
  s0+=0.007295466388744137;
  s1+=0.9927045336112559;
 }
}
if(i9<2.0955991744995117){
 if(i11<21.941892623901367){
  s0+=0.9957016267689459;
  s1+=0.004298373231054094;
 } else {
  s0+=0.125;
  s1+=0.875;
 }
} else {
 if(i8<2.809490203857422){
  s0+=0.41935483870967744;
  s1+=0.5806451612903226;
 } else {
  s0+=0.013631937682570594;
  s1+=0.9863680623174295;
 }
}
if(i13<2.9448680877685547){
 if(i6<1.9102973937988281){
  s0+=0.9951050107491318;
  s1+=0.004894989250868199;
 } else {
  s0+=0.14814814814814814;
  s1+=0.8518518518518519;
 }
} else {
 if(i13<4.481876373291016){
  s0+=0.35602094240837695;
  s1+=0.643979057591623;
 } else {
  s0+=0.014821676702176934;
  s1+=0.9851783232978231;
 }
}
if(i13<2.9785900115966797){
 if(i7<1.6634936332702637){
  s0+=0.9949660539824474;
  s1+=0.005033946017552575;
 } else {
  s0+=0.28846153846153844;
  s1+=0.7115384615384616;
 }
} else {
 if(i12<8.821407318115234){
  s0+=0.291970802919708;
  s1+=0.708029197080292;
 } else {
  s0+=0.0030257186081694403;
  s1+=0.9969742813918305;
 }
}
if(i12<5.736588478088379){
 if(i9<2.3828983306884766){
  s0+=0.9957881471163732;
  s1+=0.004211852883626836;
 } else {
  s0+=0.2364341085271318;
  s1+=0.7635658914728682;
 }
} else {
 if(i7<2.2480926513671875){
  s0+=0.34770114942528735;
  s1+=0.6522988505747126;
 } else {
  s0+=0.013037180106228875;
  s1+=0.9869628198937711;
 }
}
if(i9<2.2372922897338867){
 if(i11<22.67719268798828){
  s0+=0.9954398255237592;
  s1+=0.00456017447624083;
 } else {
  s0+=0.12994350282485875;
  s1+=0.8700564971751412;
 }
} else {
 if(i8<2.6615447998046875){
  s0+=0.4308176100628931;
  s1+=0.5691823899371069;
 } else {
  s0+=0.017848528702363725;
  s1+=0.9821514712976362;
 }
}
if(i6<1.0616941452026367){
 if(i10<3.0016698837280273){
  s0+=0.9935357687462706;
  s1+=0.006464231253729364;
 } else {
  s0+=0.125;
  s1+=0.875;
 }
} else {
 if(i12<6.661260604858398){
  s0+=0.8504273504273504;
  s1+=0.14957264957264957;
 } else {
  s0+=0.020202020202020204;
  s1+=0.9797979797979798;
 }
}
if(i8<1.9208316802978516){
 if(i7<1.6488237380981445){
  s0+=0.9947474480525916;
  s1+=0.005252551947408411;
 } else {
  s0+=0.2459016393442623;
  s1+=0.7540983606557377;
 }
} else {
 if(i8<2.607542037963867){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s0+=0.012832263978001834;
  s1+=0.9871677360219981;
 }
}
if(i8<1.8206987380981445){
 if(i9<2.080331802368164){
  s0+=0.9952448568503781;
  s1+=0.0047551431496219;
 } else {
  s0+=0.47794117647058826;
  s1+=0.5220588235294118;
 }
} else {
 if(i13<4.355199813842773){
  s0+=0.3685897435897436;
  s1+=0.6314102564102564;
 } else {
  s0+=0.01572177227251072;
  s1+=0.9842782277274893;
 }
}
if(i13<2.8801326751708984){
 if(i9<2.384824752807617){
  s0+=0.9951753081524074;
  s1+=0.0048246918475926106;
 } else {
  s0+=0.06896551724137931;
  s1+=0.9310344827586207;
 }
} else {
 if(i14<2.850727081298828){
  s0+=0.25788497217068646;
  s1+=0.7421150278293135;
 } else {
  s0+=0.009495252373813094;
  s1+=0.9905047476261869;
 }
}
if(i10<2.366262435913086){
 if(i6<1.797703742980957){
  s0+=0.9948320413436692;
  s1+=0.00516795865633075;
 } else {
  s0+=0.06103286384976526;
  s1+=0.9389671361502347;
 }
} else {
 if(i13<4.331027984619141){
  s0+=0.6009615384615384;
  s1+=0.39903846153846156;
 } else {
  s0+=0.017866004962779156;
  s1+=0.9821339950372209;
 }
}
if(i13<2.925356864929199){
 if(i11<20.709352493286133){
  s0+=0.9950079344088866;
  s1+=0.004992065591113462;
 } else {
  s0+=0.3048780487804878;
  s1+=0.6951219512195121;
 }
} else {
 if(i7<2.568282127380371){
  s0+=0.2793388429752066;
  s1+=0.7206611570247934;
 } else {
  s0+=0.0015831134564643799;
  s1+=0.9984168865435357;
 }
}
if(i14<2.00339412689209){
 if(i6<1.6106739044189453){
  s0+=0.9952506596306069;
  s1+=0.00474934036939314;
 } else {
  s0+=0.12234042553191489;
  s1+=0.8776595744680851;
 }
} else {
 if(i12<7.754186630249023){
  s0+=0.3502202643171806;
  s1+=0.6497797356828194;
 } else {
  s0+=0.008565310492505354;
  s1+=0.9914346895074947;
 }
}
if(i7<1.6513919830322266){
 if(i14<1.940725326538086){
  s0+=0.9977752689600212;
  s1+=0.002224731039978749;
 } else {
  s0+=0.3870967741935484;
  s1+=0.6129032258064516;
 }
} else {
 if(i8<2.6514711380004883){
  s0+=0.36213991769547327;
  s1+=0.6378600823045267;
 } else {
  s0+=0.018580276322058123;
  s1+=0.9814197236779418;
 }
}
if(i8<1.8746299743652344){
 if(i11<18.912586212158203){
  s0+=0.995314768377986;
  s1+=0.0046852316220139896;
 } else {
  s0+=0.4423076923076923;
  s1+=0.5576923076923077;
 }
} else {
 if(i12<8.16076946258545){
  s0+=0.2826603325415677;
  s1+=0.7173396674584323;
 } else {
  s0+=0.007511266900350526;
  s1+=0.9924887330996495;
 }
}
if(i8<1.7743005752563477){
 if(i9<2.2372922897338867){
  s0+=0.99498233915426;
  s1+=0.00501766084573994;
 } else {
  s0+=0.4084507042253521;
  s1+=0.5915492957746479;
 }
} else {
 if(i14<2.793036460876465){
  s0+=0.2920168067226891;
  s1+=0.707983193277311;
 } else {
  s0+=0.015577889447236181;
  s1+=0.9844221105527639;
 }
}
if(i8<1.7945966720581055){
 if(i12<5.776680946350098){
  s0+=0.9951287404314544;
  s1+=0.0048712595685455815;
 } else {
  s0+=0.35135135135135137;
  s1+=0.6486486486486487;
 }
} else {
 if(i7<2.2616634368896484){
  s0+=0.3076923076923077;
  s1+=0.6923076923076923;
 } else {
  s0+=0.00634765625;
  s1+=0.99365234375;
 }
}
if(i9<2.083646774291992){
 if(i9<1.3851466178894043){
  s0+=0.9968142072798752;
  s1+=0.0031857927201247205;
 } else {
  s0+=0.8285024154589372;
  s1+=0.17149758454106281;
 }
} else {
 if(i13<4.331027984619141){
  s0+=0.43626062322946174;
  s1+=0.5637393767705382;
 } else {
  s0+=0.02006532897806813;
  s1+=0.9799346710219319;
 }
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
