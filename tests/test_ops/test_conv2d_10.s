
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_9_5_3_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_i32 s0, s14, 7                                       // 000000001714: 9600870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s1, s0, 31                                      // 00000000171C: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001724: 84808200
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s6, s6, s0                                       // 00000000172C: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001730: 82070107
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001738: 84808202
	s_mul_i32 s2, s15, 27                                      // 00000000173C: 96029B0F
	s_add_u32 s6, s6, s0                                       // 000000001740: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001744: 82070107
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001750: 84828202
	s_add_u32 s2, s8, s2                                       // 000000001754: 80020208
	s_addc_u32 s3, s9, s3                                      // 000000001758: 82030309
	s_load_b512 s[16:31], s[2:3], null                         // 00000000175C: F4100401 F8000000
	s_clause 0x3                                               // 000000001764: BF850003
	s_load_b64 s[8:9], s[6:7], null                            // 000000001768: F4040203 F8000000
	s_load_b32 s12, s[6:7], 0x8                                // 000000001770: F4000303 F8000008
	s_load_b64 s[10:11], s[6:7], 0x1c                          // 000000001778: F4040283 F800001C
	s_load_b32 s13, s[6:7], 0x24                               // 000000001780: F4000343 F8000024
	s_waitcnt lgkmcnt(0)                                       // 000000001788: BF89FC07
	v_fma_f32 v0, s8, s16, 0                                   // 00000000178C: D6130000 02002008
	s_load_b32 s16, s[6:7], 0x40                               // 000000001794: F4000403 F8000040
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000179C: BF8700C1
	v_fmac_f32_e64 v0, s9, s17                                 // 0000000017A0: D52B0000 00002209
	s_clause 0x1                                               // 0000000017A8: BF850001
	s_load_b64 s[8:9], s[6:7], 0x38                            // 0000000017AC: F4040203 F8000038
	s_load_b32 s17, s[6:7], 0x13c                              // 0000000017B4: F4000443 F800013C
	v_fmac_f32_e64 v0, s12, s18                                // 0000000017BC: D52B0000 0000240C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870091
	v_fmac_f32_e64 v0, s10, s19                                // 0000000017C8: D52B0000 0000260A
	v_fmac_f32_e64 v0, s11, s20                                // 0000000017D0: D52B0000 0000280B
	s_load_b64 s[10:11], s[6:7], 0x134                         // 0000000017D8: F4040283 F8000134
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E0: BF8700A1
	v_fmac_f32_e64 v0, s13, s21                                // 0000000017E4: D52B0000 00002A0D
	s_waitcnt lgkmcnt(0)                                       // 0000000017EC: BF89FC07
	v_fmac_f32_e64 v0, s8, s22                                 // 0000000017F0: D52B0000 00002C08
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017F8: BF8700C1
	v_fmac_f32_e64 v0, s9, s23                                 // 0000000017FC: D52B0000 00002E09
	s_clause 0x1                                               // 000000001804: BF850001
	s_load_b64 s[8:9], s[6:7], 0x16c                           // 000000001808: F4040203 F800016C
	s_load_b64 s[12:13], s[6:7], 0x150                         // 000000001810: F4040303 F8000150
	v_fmac_f32_e64 v0, s16, s24                                // 000000001818: D52B0000 00003010
	s_load_b32 s24, s[6:7], 0x158                              // 000000001820: F4000603 F8000158
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001828: BF8700A1
	v_fmac_f32_e64 v0, s10, s25                                // 00000000182C: D52B0000 0000320A
	s_load_b32 s25, s[6:7], 0x174                              // 000000001834: F4000643 F8000174
	v_fmac_f32_e64 v0, s11, s26                                // 00000000183C: D52B0000 0000340B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001844: BF8700B1
	v_fmac_f32_e64 v0, s17, s27                                // 000000001848: D52B0000 00003611
	s_load_b256 s[16:23], s[2:3], 0x40                         // 000000001850: F40C0401 F8000040
	s_waitcnt lgkmcnt(0)                                       // 000000001858: BF89FC07
	v_fmac_f32_e64 v0, s12, s28                                // 00000000185C: D52B0000 0000380C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001864: BF8700C1
	v_fmac_f32_e64 v0, s13, s29                                // 000000001868: D52B0000 00003A0D
	s_clause 0x1                                               // 000000001870: BF850001
	s_load_b64 s[10:11], s[6:7], 0x284                         // 000000001874: F4040283 F8000284
	s_load_b64 s[12:13], s[6:7], 0x268                         // 00000000187C: F4040303 F8000268
	v_fmac_f32_e64 v0, s24, s30                                // 000000001884: D52B0000 00003C18
	s_load_b32 s24, s[6:7], 0x270                              // 00000000188C: F4000603 F8000270
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001894: BF870091
	v_fmac_f32_e64 v0, s8, s31                                 // 000000001898: D52B0000 00003E08
	v_fmac_f32_e64 v0, s9, s16                                 // 0000000018A0: D52B0000 00002009
	s_load_b32 s16, s[6:7], 0x28c                              // 0000000018A8: F4000403 F800028C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018B0: BF8700A1
	v_fmac_f32_e64 v0, s25, s17                                // 0000000018B4: D52B0000 00002219
	s_waitcnt lgkmcnt(0)                                       // 0000000018BC: BF89FC07
	v_fmac_f32_e64 v0, s12, s18                                // 0000000018C0: D52B0000 0000240C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000018C8: BF8700B1
	v_fmac_f32_e64 v0, s13, s19                                // 0000000018CC: D52B0000 0000260D
	s_load_b64 s[8:9], s[6:7], 0x2a0                           // 0000000018D4: F4040203 F80002A0
	s_load_b64 s[12:13], s[2:3], 0x60                          // 0000000018DC: F4040301 F8000060
	v_fmac_f32_e64 v0, s24, s20                                // 0000000018E4: D52B0000 00002818
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018EC: BF870001
	v_fmac_f32_e64 v0, s10, s21                                // 0000000018F0: D52B0000 00002A0A
	s_load_b32 s7, s[6:7], 0x2a8                               // 0000000018F8: F40001C3 F80002A8
	s_load_b32 s10, s[2:3], 0x68                               // 000000001900: F4000281 F8000068
	s_mul_i32 s2, s15, 45                                      // 000000001908: 9602AD0F
	s_mul_i32 s6, s14, 5                                       // 00000000190C: 9606850E
	s_ashr_i32 s3, s2, 31                                      // 000000001910: 86039F02
	v_fmac_f32_e64 v0, s11, s22                                // 000000001914: D52B0000 00002C0B
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000191C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001920: BF8700A9
	s_add_u32 s4, s4, s2                                       // 000000001924: 80040204
	s_addc_u32 s5, s5, s3                                      // 000000001928: 82050305
	v_fmac_f32_e64 v0, s16, s23                                // 00000000192C: D52B0000 00002E10
	s_waitcnt lgkmcnt(0)                                       // 000000001934: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001938: BF870091
	v_fmac_f32_e64 v0, s8, s12                                 // 00000000193C: D52B0000 00001808
	v_fmac_f32_e64 v0, s9, s13                                 // 000000001944: D52B0000 00001A09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000194C: BF870141
	v_fmac_f32_e64 v0, s7, s10                                 // 000000001950: D52B0000 00001407
	s_ashr_i32 s7, s6, 31                                      // 000000001958: 86079F06
	v_mov_b32_e32 v1, 0                                        // 00000000195C: 7E020280
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001960: 84828206
	v_max_f32_e32 v0, 0, v0                                    // 000000001964: 20000080
	s_add_u32 s2, s4, s2                                       // 000000001968: 80020204
	s_addc_u32 s3, s5, s3                                      // 00000000196C: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001970: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001974: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001978: DC6A0000 00000001
	s_nop 0                                                    // 000000001980: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001984: BFB60003
	s_endpgm                                                   // 000000001988: BFB00000
