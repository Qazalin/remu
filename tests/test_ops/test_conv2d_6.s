
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_10_5_3_2_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s14, 7                                       // 000000001714: 9608870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001720: BF870009
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000172C: 80080806
	s_addc_u32 s7, s7, s9                                      // 000000001730: 82070907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_mul_i32 s6, s15, 18                                      // 000000001738: 9606920F
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000173C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s8, s8, s2                                       // 000000001744: 80080208
	s_addc_u32 s9, s7, s3                                      // 000000001748: 82090307
	s_ashr_i32 s7, s6, 31                                      // 00000000174C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001750: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s0, s0, s6                                       // 000000001758: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000175C: 82010701
	s_load_b512 s[16:31], s[0:1], null                         // 000000001760: F4100400 F8000000
	s_clause 0x3                                               // 000000001768: BF850003
	s_load_b64 s[6:7], s[8:9], null                            // 00000000176C: F4040184 F8000000
	s_load_b32 s33, s[8:9], 0x8                                // 000000001774: F4000844 F8000008
	s_load_b64 s[10:11], s[8:9], 0x1c                          // 00000000177C: F4040284 F800001C
	s_load_b32 s34, s[8:9], 0x24                               // 000000001784: F4000884 F8000024
	s_waitcnt lgkmcnt(0)                                       // 00000000178C: BF89FC07
	v_fma_f32 v0, s6, s16, 0                                   // 000000001790: D6130000 02002006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001798: BF870001
	v_fmac_f32_e64 v0, s7, s17                                 // 00000000179C: D52B0000 00002207
	s_clause 0x3                                               // 0000000017A4: BF850003
	s_load_b64 s[6:7], s[8:9], 0x150                           // 0000000017A8: F4040184 F8000150
	s_load_b64 s[12:13], s[8:9], 0x134                         // 0000000017B0: F4040304 F8000134
	s_load_b32 s16, s[8:9], 0x13c                              // 0000000017B8: F4000404 F800013C
	s_load_b32 s17, s[8:9], 0x158                              // 0000000017C0: F4000444 F8000158
	v_fmac_f32_e64 v0, s33, s18                                // 0000000017C8: D52B0000 00002421
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D0: BF870091
	v_fmac_f32_e64 v0, s10, s19                                // 0000000017D4: D52B0000 0000260A
	v_fmac_f32_e64 v0, s11, s20                                // 0000000017DC: D52B0000 0000280B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E4: BF8700A1
	v_fmac_f32_e64 v0, s34, s21                                // 0000000017E8: D52B0000 00002A22
	s_waitcnt lgkmcnt(0)                                       // 0000000017F0: BF89FC07
	v_fmac_f32_e64 v0, s12, s22                                // 0000000017F4: D52B0000 00002C0C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017FC: BF870001
	v_fmac_f32_e64 v0, s13, s23                                // 000000001800: D52B0000 00002E0D
	s_clause 0x1                                               // 000000001808: BF850001
	s_load_b64 s[10:11], s[8:9], 0x284                         // 00000000180C: F4040284 F8000284
	s_load_b64 s[12:13], s[8:9], 0x268                         // 000000001814: F4040304 F8000268
	s_load_b64 s[0:1], s[0:1], 0x40                            // 00000000181C: F4040000 F8000040
	v_fmac_f32_e64 v0, s16, s24                                // 000000001824: D52B0000 00003010
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000182C: BF8700C1
	v_fmac_f32_e64 v0, s6, s25                                 // 000000001830: D52B0000 00003206
	s_clause 0x1                                               // 000000001838: BF850001
	s_load_b32 s6, s[8:9], 0x270                               // 00000000183C: F4000184 F8000270
	s_load_b32 s8, s[8:9], 0x28c                               // 000000001844: F4000204 F800028C
	v_fmac_f32_e64 v0, s7, s26                                 // 00000000184C: D52B0000 00003407
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001854: BF8700A1
	v_fmac_f32_e64 v0, s17, s27                                // 000000001858: D52B0000 00003611
	s_waitcnt lgkmcnt(0)                                       // 000000001860: BF89FC07
	v_fmac_f32_e64 v0, s12, s28                                // 000000001864: D52B0000 0000380C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000186C: BF870091
	v_fmac_f32_e64 v0, s13, s29                                // 000000001870: D52B0000 00003A0D
	v_fmac_f32_e64 v0, s6, s30                                 // 000000001878: D52B0000 00003C06
	s_mul_i32 s6, s15, 50                                      // 000000001880: 9606B20F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001884: BF870099
	s_ashr_i32 s7, s6, 31                                      // 000000001888: 86079F06
	v_fmac_f32_e64 v0, s10, s31                                // 00000000188C: D52B0000 00003E0A
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001894: 84868206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001898: BF8700A9
	s_add_u32 s4, s4, s6                                       // 00000000189C: 80040604
	s_addc_u32 s5, s5, s7                                      // 0000000018A0: 82050705
	v_fmac_f32_e64 v0, s11, s0                                 // 0000000018A4: D52B0000 0000000B
	s_mul_i32 s0, s14, 5                                       // 0000000018AC: 9600850E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000018B0: BF870141
	v_fmac_f32_e64 v0, s8, s1                                  // 0000000018B4: D52B0000 00000208
	s_ashr_i32 s1, s0, 31                                      // 0000000018BC: 86019F00
	v_mov_b32_e32 v1, 0                                        // 0000000018C0: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000018C4: 84808200
	v_max_f32_e32 v0, 0, v0                                    // 0000000018C8: 20000080
	s_add_u32 s0, s4, s0                                       // 0000000018CC: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000018D0: 82010105
	s_add_u32 s0, s0, s2                                       // 0000000018D4: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000018D8: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000018DC: DC6A0000 00000001
	s_nop 0                                                    // 0000000018E4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018E8: BFB60003
	s_endpgm                                                   // 0000000018EC: BFB00000
