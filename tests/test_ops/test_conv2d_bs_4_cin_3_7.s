
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_10_3_3_2_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_hi_i32 s2, s13, 0x55555556                           // 000000001714: 9702FF0D 55555556
	s_mul_i32 s0, s15, 0xe7                                    // 00000000171C: 9600FF0F 000000E7
	s_lshr_b32 s3, s2, 31                                      // 000000001724: 85039F02
	s_ashr_i32 s1, s0, 31                                      // 000000001728: 86019F00
	s_add_i32 s10, s2, s3                                      // 00000000172C: 810A0302
	s_lshl_b64 s[2:3], s[0:1], 2                               // 000000001730: 84828200
	s_mul_i32 s0, s10, 3                                       // 000000001734: 9600830A
	s_mul_i32 s10, s10, 7                                      // 000000001738: 960A870A
	s_sub_i32 s12, s13, s0                                     // 00000000173C: 818C000D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s1, s6, s2                                       // 000000001744: 80010206
	s_addc_u32 s6, s7, s3                                      // 000000001748: 82060307
	s_ashr_i32 s11, s10, 31                                    // 00000000174C: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001754: 8482820A
	s_add_u32 s1, s1, s2                                       // 000000001758: 80010201
	s_addc_u32 s7, s6, s3                                      // 00000000175C: 82070306
	s_ashr_i32 s13, s12, 31                                    // 000000001760: 860D9F0C
	s_mul_i32 s6, s14, 30                                      // 000000001764: 96069E0E
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000001768: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704B9
	s_add_u32 s12, s1, s2                                      // 000000001770: 800C0201
	s_addc_u32 s13, s7, s3                                     // 000000001774: 820D0307
	s_ashr_i32 s7, s6, 31                                      // 000000001778: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000177C: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s34, s8, s6                                      // 000000001784: 80220608
	s_addc_u32 s35, s9, s7                                     // 000000001788: 82230709
	s_load_b512 s[16:31], s[34:35], null                       // 00000000178C: F4100411 F8000000
	s_clause 0x2                                               // 000000001794: BF850002
	s_load_b128 s[8:11], s[12:13], null                        // 000000001798: F4080206 F8000000
	s_load_b32 s1, s[12:13], 0x10                              // 0000000017A0: F4000046 F8000010
	s_load_b128 s[36:39], s[12:13], 0x1c                       // 0000000017A8: F4080906 F800001C
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_fma_f32 v0, s8, s16, 0                                   // 0000000017B4: D6130000 02002008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017BC: BF870091
	v_fmac_f32_e64 v0, s9, s17                                 // 0000000017C0: D52B0000 00002209
	v_fmac_f32_e64 v0, s10, s18                                // 0000000017C8: D52B0000 0000240A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017D0: BF8700A1
	v_fmac_f32_e64 v0, s11, s19                                // 0000000017D4: D52B0000 0000260B
	s_load_b128 s[8:11], s[12:13], 0x134                       // 0000000017DC: F4080206 F8000134
	v_fmac_f32_e64 v0, s1, s20                                 // 0000000017E4: D52B0000 00002801
	s_load_b32 s1, s[12:13], 0x2c                              // 0000000017EC: F4000046 F800002C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F4: BF870091
	v_fmac_f32_e64 v0, s36, s21                                // 0000000017F8: D52B0000 00002A24
	v_fmac_f32_e64 v0, s37, s22                                // 000000001800: D52B0000 00002C25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001808: BF8700A1
	v_fmac_f32_e64 v0, s38, s23                                // 00000000180C: D52B0000 00002E26
	s_load_b256 s[16:23], s[34:35], 0x40                       // 000000001814: F40C0411 F8000040
	v_fmac_f32_e64 v0, s39, s24                                // 00000000181C: D52B0000 00003027
	s_load_b128 s[36:39], s[12:13], 0x150                      // 000000001824: F4080906 F8000150
	s_waitcnt lgkmcnt(0)                                       // 00000000182C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001830: BF8700A1
	v_fmac_f32_e64 v0, s1, s25                                 // 000000001834: D52B0000 00003201
	s_load_b32 s1, s[12:13], 0x144                             // 00000000183C: F4000046 F8000144
	v_fmac_f32_e64 v0, s8, s26                                 // 000000001844: D52B0000 00003408
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000184C: BF870091
	v_fmac_f32_e64 v0, s9, s27                                 // 000000001850: D52B0000 00003609
	v_fmac_f32_e64 v0, s10, s28                                // 000000001858: D52B0000 0000380A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001860: BF8700B1
	v_fmac_f32_e64 v0, s11, s29                                // 000000001864: D52B0000 00003A0B
	s_load_b128 s[8:11], s[12:13], 0x268                       // 00000000186C: F4080206 F8000268
	s_waitcnt lgkmcnt(0)                                       // 000000001874: BF89FC07
	v_fmac_f32_e64 v0, s1, s30                                 // 000000001878: D52B0000 00003C01
	s_load_b32 s1, s[12:13], 0x160                             // 000000001880: F4000046 F8000160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001888: BF870091
	v_fmac_f32_e64 v0, s36, s31                                // 00000000188C: D52B0000 00003E24
	v_fmac_f32_e64 v0, s37, s16                                // 000000001894: D52B0000 00002025
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000189C: BF870091
	v_fmac_f32_e64 v0, s38, s17                                // 0000000018A0: D52B0000 00002226
	v_fmac_f32_e64 v0, s39, s18                                // 0000000018A8: D52B0000 00002427
	s_waitcnt lgkmcnt(0)                                       // 0000000018B0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000018B4: BF8700C1
	v_fmac_f32_e64 v0, s1, s19                                 // 0000000018B8: D52B0000 00002601
	s_load_b32 s1, s[12:13], 0x278                             // 0000000018C0: F4000046 F8000278
	s_load_b128 s[16:19], s[34:35], 0x60                       // 0000000018C8: F4080411 F8000060
	s_load_b128 s[24:27], s[12:13], 0x284                      // 0000000018D0: F4080606 F8000284
	v_fmac_f32_e64 v0, s8, s20                                 // 0000000018D8: D52B0000 00002808
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018E0: BF8700A1
	v_fmac_f32_e64 v0, s9, s21                                 // 0000000018E4: D52B0000 00002A09
	s_load_b64 s[8:9], s[34:35], 0x70                          // 0000000018EC: F4040211 F8000070
	v_fmac_f32_e64 v0, s10, s22                                // 0000000018F4: D52B0000 00002C0A
	s_mul_i32 s10, s15, 0xb4                                   // 0000000018FC: 960AFF0F 000000B4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001904: BF8704A1
	v_fmac_f32_e64 v0, s11, s23                                // 000000001908: D52B0000 00002E0B
	s_ashr_i32 s11, s10, 31                                    // 000000001910: 860B9F0A
	s_lshl_b64 s[10:11], s[10:11], 2                           // 000000001914: 848A820A
	s_waitcnt lgkmcnt(0)                                       // 000000001918: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000191C: BF870001
	v_fmac_f32_e64 v0, s1, s16                                 // 000000001920: D52B0000 00002001
	s_load_b32 s1, s[12:13], 0x294                             // 000000001928: F4000046 F8000294
	s_add_u32 s4, s4, s10                                      // 000000001930: 80040A04
	s_addc_u32 s5, s5, s11                                     // 000000001934: 82050B05
	s_add_u32 s4, s4, s6                                       // 000000001938: 80040604
	v_fmac_f32_e64 v0, s24, s17                                // 00000000193C: D52B0000 00002218
	s_addc_u32 s5, s5, s7                                      // 000000001944: 82050705
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001948: BF870091
	v_fmac_f32_e64 v0, s25, s18                                // 00000000194C: D52B0000 00002419
	v_fmac_f32_e64 v0, s26, s19                                // 000000001954: D52B0000 0000261A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000195C: BF8700A1
	v_fmac_f32_e64 v0, s27, s8                                 // 000000001960: D52B0000 0000101B
	s_waitcnt lgkmcnt(0)                                       // 000000001968: BF89FC07
	v_fmac_f32_e64 v0, s1, s9                                  // 00000000196C: D52B0000 00001201
	s_ashr_i32 s1, s0, 31                                      // 000000001974: 86019F00
	v_mov_b32_e32 v1, 0                                        // 000000001978: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000197C: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001980: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001984: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001988: 80000004
	s_addc_u32 s1, s5, s1                                      // 00000000198C: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001990: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001994: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001998: DC6A0000 00000001
	s_nop 0                                                    // 0000000019A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000019A4: BFB60003
	s_endpgm                                                   // 0000000019A8: BFB00000
