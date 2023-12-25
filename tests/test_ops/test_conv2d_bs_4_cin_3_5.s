
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_10_6_3_2_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s8, s13, 0x2aaaaaab                           // 000000001714: 9708FF0D 2AAAAAAB
	s_mul_i32 s2, s15, 0xe7                                    // 00000000171C: 9602FF0F 000000E7
	s_lshr_b32 s9, s8, 31                                      // 000000001724: 85099F08
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_add_i32 s9, s8, s9                                       // 00000000172C: 81090908
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_mul_i32 s8, s9, 6                                        // 000000001734: 96088609
	s_mul_i32 s10, s9, 7                                       // 000000001738: 960A8709
	s_sub_i32 s12, s13, s8                                     // 00000000173C: 818C080D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001744: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001748: 82070307
	s_ashr_i32 s11, s10, 31                                    // 00000000174C: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001754: 8482820A
	s_add_u32 s9, s6, s2                                       // 000000001758: 80090206
	s_addc_u32 s3, s7, s3                                      // 00000000175C: 82030307
	s_ashr_i32 s13, s12, 31                                    // 000000001760: 860D9F0C
	s_mul_i32 s2, s14, 12                                      // 000000001764: 96028C0E
	s_lshl_b64 s[6:7], s[12:13], 2                             // 000000001768: 8486820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704B9
	s_add_u32 s10, s9, s6                                      // 000000001770: 800A0609
	s_addc_u32 s11, s3, s7                                     // 000000001774: 820B0703
	s_ashr_i32 s3, s2, 31                                      // 000000001778: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000177C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s0, s0, s2                                       // 000000001784: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001788: 82010301
	s_load_b256 s[16:23], s[0:1], null                         // 00000000178C: F40C0400 F8000000
	s_clause 0x3                                               // 000000001794: BF850003
	s_load_b64 s[2:3], s[10:11], null                          // 000000001798: F4040085 F8000000
	s_load_b64 s[12:13], s[10:11], 0x1c                        // 0000000017A0: F4040305 F800001C
	s_load_b64 s[24:25], s[10:11], 0x134                       // 0000000017A8: F4040605 F8000134
	s_load_b64 s[26:27], s[10:11], 0x150                       // 0000000017B0: F4040685 F8000150
	s_waitcnt lgkmcnt(0)                                       // 0000000017B8: BF89FC07
	v_fma_f32 v0, s2, s16, 0                                   // 0000000017BC: D6130000 02002002
	s_mul_i32 s16, s15, 0x168                                  // 0000000017C4: 9610FF0F 00000168
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017CC: BF8704B1
	v_fmac_f32_e64 v0, s3, s17                                 // 0000000017D0: D52B0000 00002203
	s_load_b128 s[0:3], s[0:1], 0x20                           // 0000000017D8: F4080000 F8000020
	s_ashr_i32 s17, s16, 31                                    // 0000000017E0: 86119F10
	s_lshl_b64 s[16:17], s[16:17], 2                           // 0000000017E4: 84908210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000017E8: BF8700B1
	v_fmac_f32_e64 v0, s12, s18                                // 0000000017EC: D52B0000 0000240C
	s_add_u32 s4, s4, s16                                      // 0000000017F4: 80041004
	s_addc_u32 s5, s5, s17                                     // 0000000017F8: 82051105
	v_fmac_f32_e64 v0, s13, s19                                // 0000000017FC: D52B0000 0000260D
	s_clause 0x1                                               // 000000001804: BF850001
	s_load_b64 s[12:13], s[10:11], 0x268                       // 000000001808: F4040305 F8000268
	s_load_b64 s[10:11], s[10:11], 0x284                       // 000000001810: F4040285 F8000284
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001818: BF870091
	v_fmac_f32_e64 v0, s24, s20                                // 00000000181C: D52B0000 00002818
	v_fmac_f32_e64 v0, s25, s21                                // 000000001824: D52B0000 00002A19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000182C: BF870091
	v_fmac_f32_e64 v0, s26, s22                                // 000000001830: D52B0000 00002C1A
	v_fmac_f32_e64 v0, s27, s23                                // 000000001838: D52B0000 00002E1B
	s_waitcnt lgkmcnt(0)                                       // 000000001840: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001844: BF8700A1
	v_fmac_f32_e64 v0, s12, s0                                 // 000000001848: D52B0000 0000000C
	s_mul_i32 s0, s14, 60                                      // 000000001850: 9600BC0E
	v_fmac_f32_e64 v0, s13, s1                                 // 000000001854: D52B0000 0000020D
	s_ashr_i32 s1, s0, 31                                      // 00000000185C: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001860: BF870099
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001864: 84808200
	v_fmac_f32_e64 v0, s10, s2                                 // 000000001868: D52B0000 0000040A
	s_add_u32 s2, s4, s0                                       // 000000001870: 80020004
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001874: BF870001
	v_fmac_f32_e64 v0, s11, s3                                 // 000000001878: D52B0000 0000060B
	s_addc_u32 s3, s5, s1                                      // 000000001880: 82030105
	s_ashr_i32 s9, s8, 31                                      // 000000001884: 86099F08
	v_mov_b32_e32 v1, 0                                        // 000000001888: 7E020280
	s_lshl_b64 s[0:1], s[8:9], 2                               // 00000000188C: 84808208
	v_max_f32_e32 v0, 0, v0                                    // 000000001890: 20000080
	s_add_u32 s0, s2, s0                                       // 000000001894: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001898: 82010103
	s_add_u32 s0, s0, s6                                       // 00000000189C: 80000600
	s_addc_u32 s1, s1, s7                                      // 0000000018A0: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 0000000018A4: DC6A0000 00000001
	s_nop 0                                                    // 0000000018AC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018B0: BFB60003
	s_endpgm                                                   // 0000000018B4: BFB00000
