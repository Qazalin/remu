
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_7_3_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_mul_i32 s8, s2, 15                                       // 00000000171C: 96088F02
	s_lshl_b64 s[24:25], s[14:15], 2                           // 000000001720: 8498820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s24                                      // 000000001728: 80061806
	s_addc_u32 s7, s7, s25                                     // 00000000172C: 82071907
	s_ashr_i32 s9, s8, 31                                      // 000000001730: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001738: 84888208
	s_add_u32 s0, s0, s8                                       // 00000000173C: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001740: 82010901
	s_load_b256 s[8:15], s[0:1], null                          // 000000001744: F40C0200 F8000000
	s_clause 0x2                                               // 00000000174C: BF850002
	s_load_b128 s[16:19], s[6:7], null                         // 000000001750: F4080403 F8000000
	s_load_b32 s3, s[6:7], 0x10                                // 000000001758: F40000C3 F8000010
	s_load_b128 s[20:23], s[6:7], 0x2c                         // 000000001760: F4080503 F800002C
	s_waitcnt lgkmcnt(0)                                       // 000000001768: BF89FC07
	v_fma_f32 v0, s16, s8, 0                                   // 00000000176C: D6130000 02001010
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001774: BF870091
	v_fmac_f32_e64 v0, s17, s9                                 // 000000001778: D52B0000 00001211
	v_fmac_f32_e64 v0, s18, s10                                // 000000001780: D52B0000 00001412
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001788: BF8700B1
	v_fmac_f32_e64 v0, s19, s11                                // 00000000178C: D52B0000 00001613
	s_load_b128 s[8:11], s[0:1], 0x20                          // 000000001794: F4080200 F8000020
	s_load_b128 s[16:19], s[6:7], 0x58                         // 00000000179C: F4080403 F8000058
	v_fmac_f32_e64 v0, s3, s12                                 // 0000000017A4: D52B0000 00001803
	s_load_b32 s3, s[6:7], 0x3c                                // 0000000017AC: F40000C3 F800003C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017B4: BF8700A1
	v_fmac_f32_e64 v0, s20, s13                                // 0000000017B8: D52B0000 00001A14
	s_load_b64 s[12:13], s[0:1], 0x30                          // 0000000017C0: F4040300 F8000030
	v_fmac_f32_e64 v0, s21, s14                                // 0000000017C8: D52B0000 00001C15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017D0: BF8700A1
	v_fmac_f32_e64 v0, s22, s15                                // 0000000017D4: D52B0000 00001E16
	s_waitcnt lgkmcnt(0)                                       // 0000000017DC: BF89FC07
	v_fmac_f32_e64 v0, s23, s8                                 // 0000000017E0: D52B0000 00001017
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017E8: BF8700C1
	v_fmac_f32_e64 v0, s3, s9                                  // 0000000017EC: D52B0000 00001203
	s_load_b32 s3, s[6:7], 0x68                                // 0000000017F4: F40000C3 F8000068
	s_load_b32 s1, s[0:1], 0x38                                // 0000000017FC: F4000040 F8000038
	s_mul_i32 s0, s2, 7                                        // 000000001804: 96008702
	v_fmac_f32_e64 v0, s16, s10                                // 000000001808: D52B0000 00001410
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001810: BF870091
	v_fmac_f32_e64 v0, s17, s11                                // 000000001814: D52B0000 00001611
	v_fmac_f32_e64 v0, s18, s12                                // 00000000181C: D52B0000 00001812
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001824: BF8700A1
	v_fmac_f32_e64 v0, s19, s13                                // 000000001828: D52B0000 00001A13
	s_waitcnt lgkmcnt(0)                                       // 000000001830: BF89FC07
	v_fmac_f32_e64 v0, s3, s1                                  // 000000001834: D52B0000 00000203
	s_ashr_i32 s1, s0, 31                                      // 00000000183C: 86019F00
	v_mov_b32_e32 v1, 0                                        // 000000001840: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001844: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001848: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 00000000184C: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001850: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001854: 82010105
	s_add_u32 s0, s0, s24                                      // 000000001858: 80001800
	s_addc_u32 s1, s1, s25                                     // 00000000185C: 82011901
	global_store_b32 v1, v0, s[0:1]                            // 000000001860: DC6A0000 00000001
	s_nop 0                                                    // 000000001868: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000186C: BFB60003
	s_endpgm                                                   // 000000001870: BFB00000
