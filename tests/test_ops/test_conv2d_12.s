
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_9_3_3_3_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_i32 s0, s14, 7                                       // 000000001714: 9600870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s1, s0, 31                                      // 00000000171C: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001720: BF870009
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001724: 84808200
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s10, s6, s0                                      // 00000000172C: 800A0006
	s_addc_u32 s7, s7, s1                                      // 000000001730: 82070107
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_mul_i32 s6, s15, 45                                      // 000000001738: 9606AD0F
	s_lshl_b64 s[0:1], s[2:3], 2                               // 00000000173C: 84808202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s2, s10, s0                                      // 000000001744: 8002000A
	s_addc_u32 s3, s7, s1                                      // 000000001748: 82030107
	s_ashr_i32 s7, s6, 31                                      // 00000000174C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001750: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s6, s8, s6                                       // 000000001758: 80060608
	s_addc_u32 s7, s9, s7                                      // 00000000175C: 82070709
	s_load_b512 s[36:51], s[6:7], null                         // 000000001760: F4100903 F8000000
	s_clause 0x2                                               // 000000001768: BF850002
	s_load_b128 s[8:11], s[2:3], null                          // 00000000176C: F4080201 F8000000
	s_load_b32 s12, s[2:3], 0x10                               // 000000001774: F4000301 F8000010
	s_load_b128 s[16:19], s[2:3], 0x1c                         // 00000000177C: F4080401 F800001C
	s_waitcnt lgkmcnt(0)                                       // 000000001784: BF89FC07
	v_fma_f32 v0, s8, s36, 0                                   // 000000001788: D6130000 02004808
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001790: BF870091
	v_fmac_f32_e64 v0, s9, s37                                 // 000000001794: D52B0000 00004A09
	v_fmac_f32_e64 v0, s10, s38                                // 00000000179C: D52B0000 00004C0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017A4: BF8700C1
	v_fmac_f32_e64 v0, s11, s39                                // 0000000017A8: D52B0000 00004E0B
	s_clause 0x1                                               // 0000000017B0: BF850001
	s_load_b128 s[8:11], s[2:3], 0x38                          // 0000000017B4: F4080201 F8000038
	s_load_b128 s[36:39], s[2:3], 0x134                        // 0000000017BC: F4080901 F8000134
	v_fmac_f32_e64 v0, s12, s40                                // 0000000017C4: D52B0000 0000500C
	s_load_b32 s12, s[2:3], 0x2c                               // 0000000017CC: F4000301 F800002C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D4: BF870091
	v_fmac_f32_e64 v0, s16, s41                                // 0000000017D8: D52B0000 00005210
	v_fmac_f32_e64 v0, s17, s42                                // 0000000017E0: D52B0000 00005411
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E8: BF870091
	v_fmac_f32_e64 v0, s18, s43                                // 0000000017EC: D52B0000 00005612
	v_fmac_f32_e64 v0, s19, s44                                // 0000000017F4: D52B0000 00005813
	s_load_b512 s[16:31], s[6:7], 0x40                         // 0000000017FC: F4100403 F8000040
	s_waitcnt lgkmcnt(0)                                       // 000000001804: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001808: BF8700A1
	v_fmac_f32_e64 v0, s12, s45                                // 00000000180C: D52B0000 00005A0C
	s_load_b32 s12, s[2:3], 0x48                               // 000000001814: F4000301 F8000048
	v_fmac_f32_e64 v0, s8, s46                                 // 00000000181C: D52B0000 00005C08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001824: BF870091
	v_fmac_f32_e64 v0, s9, s47                                 // 000000001828: D52B0000 00005E09
	v_fmac_f32_e64 v0, s10, s48                                // 000000001830: D52B0000 0000600A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001838: BF8700B1
	v_fmac_f32_e64 v0, s11, s49                                // 00000000183C: D52B0000 0000620B
	s_load_b128 s[8:11], s[2:3], 0x150                         // 000000001844: F4080201 F8000150
	s_waitcnt lgkmcnt(0)                                       // 00000000184C: BF89FC07
	v_fmac_f32_e64 v0, s12, s50                                // 000000001850: D52B0000 0000640C
	s_load_b32 s12, s[2:3], 0x144                              // 000000001858: F4000301 F8000144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001860: BF870091
	v_fmac_f32_e64 v0, s36, s51                                // 000000001864: D52B0000 00006624
	v_fmac_f32_e64 v0, s37, s16                                // 00000000186C: D52B0000 00002025
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001874: BF870091
	v_fmac_f32_e64 v0, s38, s17                                // 000000001878: D52B0000 00002226
	v_fmac_f32_e64 v0, s39, s18                                // 000000001880: D52B0000 00002427
	s_load_b128 s[36:39], s[2:3], 0x16c                        // 000000001888: F4080901 F800016C
	s_waitcnt lgkmcnt(0)                                       // 000000001890: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001894: BF8700A1
	v_fmac_f32_e64 v0, s12, s19                                // 000000001898: D52B0000 0000260C
	s_load_b32 s12, s[2:3], 0x160                              // 0000000018A0: F4000301 F8000160
	v_fmac_f32_e64 v0, s8, s20                                 // 0000000018A8: D52B0000 00002808
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018B0: BF870091
	v_fmac_f32_e64 v0, s9, s21                                 // 0000000018B4: D52B0000 00002A09
	v_fmac_f32_e64 v0, s10, s22                                // 0000000018BC: D52B0000 00002C0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000018C4: BF8700C1
	v_fmac_f32_e64 v0, s11, s23                                // 0000000018C8: D52B0000 00002E0B
	s_load_b128 s[8:11], s[2:3], 0x268                         // 0000000018D0: F4080201 F8000268
	s_load_b256 s[16:23], s[6:7], 0x80                         // 0000000018D8: F40C0403 F8000080
	s_waitcnt lgkmcnt(0)                                       // 0000000018E0: BF89FC07
	v_fmac_f32_e64 v0, s12, s24                                // 0000000018E4: D52B0000 0000300C
	s_load_b32 s12, s[2:3], 0x17c                              // 0000000018EC: F4000301 F800017C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F4: BF870091
	v_fmac_f32_e64 v0, s36, s25                                // 0000000018F8: D52B0000 00003224
	v_fmac_f32_e64 v0, s37, s26                                // 000000001900: D52B0000 00003425
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001908: BF8700A1
	v_fmac_f32_e64 v0, s38, s27                                // 00000000190C: D52B0000 00003626
	s_load_b128 s[24:27], s[2:3], 0x284                        // 000000001914: F4080601 F8000284
	v_fmac_f32_e64 v0, s39, s28                                // 00000000191C: D52B0000 00003827
	s_waitcnt lgkmcnt(0)                                       // 000000001924: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001928: BF8700A1
	v_fmac_f32_e64 v0, s12, s29                                // 00000000192C: D52B0000 00003A0C
	s_load_b32 s12, s[2:3], 0x278                              // 000000001934: F4000301 F8000278
	v_fmac_f32_e64 v0, s8, s30                                 // 00000000193C: D52B0000 00003C08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001944: BF870091
	v_fmac_f32_e64 v0, s9, s31                                 // 000000001948: D52B0000 00003E09
	v_fmac_f32_e64 v0, s10, s16                                // 000000001950: D52B0000 0000200A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001958: BF8700A1
	v_fmac_f32_e64 v0, s11, s17                                // 00000000195C: D52B0000 0000220B
	s_waitcnt lgkmcnt(0)                                       // 000000001964: BF89FC07
	v_fmac_f32_e64 v0, s12, s18                                // 000000001968: D52B0000 0000240C
	s_load_b32 s12, s[2:3], 0x294                              // 000000001970: F4000301 F8000294
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001978: BF8700B1
	v_fmac_f32_e64 v0, s24, s19                                // 00000000197C: D52B0000 00002618
	s_load_b128 s[8:11], s[2:3], 0x2a0                         // 000000001984: F4080201 F80002A0
	s_load_b128 s[16:19], s[6:7], 0xa0                         // 00000000198C: F4080403 F80000A0
	v_fmac_f32_e64 v0, s25, s20                                // 000000001994: D52B0000 00002819
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000199C: BF870091
	v_fmac_f32_e64 v0, s26, s21                                // 0000000019A0: D52B0000 00002A1A
	v_fmac_f32_e64 v0, s27, s22                                // 0000000019A8: D52B0000 00002C1B
	s_waitcnt lgkmcnt(0)                                       // 0000000019B0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019B4: BF870001
	v_fmac_f32_e64 v0, s12, s23                                // 0000000019B8: D52B0000 00002E0C
	s_load_b32 s12, s[2:3], 0x2b0                              // 0000000019C0: F4000301 F80002B0
	s_load_b32 s7, s[6:7], 0xb0                                // 0000000019C8: F40001C3 F80000B0
	s_mul_i32 s2, s15, 27                                      // 0000000019D0: 96029B0F
	s_mul_i32 s6, s14, 3                                       // 0000000019D4: 9606830E
	s_ashr_i32 s3, s2, 31                                      // 0000000019D8: 86039F02
	v_fmac_f32_e64 v0, s8, s16                                 // 0000000019DC: D52B0000 00002008
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000019E4: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019E8: BF8700A9
	s_add_u32 s4, s4, s2                                       // 0000000019EC: 80040204
	s_addc_u32 s5, s5, s3                                      // 0000000019F0: 82050305
	v_fmac_f32_e64 v0, s9, s17                                 // 0000000019F4: D52B0000 00002209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019FC: BF870091
	v_fmac_f32_e64 v0, s10, s18                                // 000000001A00: D52B0000 0000240A
	v_fmac_f32_e64 v0, s11, s19                                // 000000001A08: D52B0000 0000260B
	s_waitcnt lgkmcnt(0)                                       // 000000001A10: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001A14: BF870141
	v_fmac_f32_e64 v0, s12, s7                                 // 000000001A18: D52B0000 00000E0C
	s_ashr_i32 s7, s6, 31                                      // 000000001A20: 86079F06
	v_mov_b32_e32 v1, 0                                        // 000000001A24: 7E020280
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001A28: 84828206
	v_max_f32_e32 v0, 0, v0                                    // 000000001A2C: 20000080
	s_add_u32 s2, s4, s2                                       // 000000001A30: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001A34: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001A38: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001A3C: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001A40: DC6A0000 00000001
	s_nop 0                                                    // 000000001A48: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A4C: BFB60003
	s_endpgm                                                   // 000000001A50: BFB00000
