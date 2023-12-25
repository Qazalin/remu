
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_6_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s15, 10                                      // 000000001714: 96088A0F
	s_mov_b32 s2, s15                                          // 000000001718: BE82000F
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s3, s6, s8                                       // 00000000172C: 80030806
	s_addc_u32 s8, s7, s9                                      // 000000001730: 82080907
	s_ashr_i32 s15, s14, 31                                    // 000000001734: 860F9F0E
	s_lshl_b64 s[6:7], s[14:15], 2                             // 000000001738: 8486820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000173C: BF870009
	s_add_u32 s16, s3, s6                                      // 000000001740: 80100603
	s_addc_u32 s17, s8, s7                                     // 000000001744: 82110708
	s_load_b256 s[8:15], s[0:1], null                          // 000000001748: F40C0200 F8000000
	s_clause 0x3                                               // 000000001750: BF850003
	s_load_b64 s[18:19], s[16:17], 0x2c                        // 000000001754: F4040488 F800002C
	s_load_b32 s3, s[16:17], 0x34                              // 00000000175C: F40000C8 F8000034
	s_load_b64 s[20:21], s[16:17], 0x54                        // 000000001764: F4040508 F8000054
	s_load_b32 s22, s[16:17], 0x5c                             // 00000000176C: F4000588 F800005C
	s_waitcnt lgkmcnt(0)                                       // 000000001774: BF89FC07
	v_fma_f32 v0, s18, s8, 0                                   // 000000001778: D6130000 02001012
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001780: BF8700A1
	v_fmac_f32_e64 v0, s19, s9                                 // 000000001784: D52B0000 00001213
	s_load_b64 s[8:9], s[16:17], 0x7c                          // 00000000178C: F4040208 F800007C
	v_fmac_f32_e64 v0, s3, s10                                 // 000000001794: D52B0000 00001403
	s_load_b32 s3, s[16:17], 0x84                              // 00000000179C: F40000C8 F8000084
	s_load_b32 s1, s[0:1], 0x20                                // 0000000017A4: F4000040 F8000020
	s_mul_i32 s0, s2, 6                                        // 0000000017AC: 96008602
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B0: BF870091
	v_fmac_f32_e64 v0, s20, s11                                // 0000000017B4: D52B0000 00001614
	v_fmac_f32_e64 v0, s21, s12                                // 0000000017BC: D52B0000 00001815
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017C4: BF8700A1
	v_fmac_f32_e64 v0, s22, s13                                // 0000000017C8: D52B0000 00001A16
	s_waitcnt lgkmcnt(0)                                       // 0000000017D0: BF89FC07
	v_fmac_f32_e64 v0, s8, s14                                 // 0000000017D4: D52B0000 00001C08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017DC: BF870091
	v_fmac_f32_e64 v0, s9, s15                                 // 0000000017E0: D52B0000 00001E09
	v_fmac_f32_e64 v0, s3, s1                                  // 0000000017E8: D52B0000 00000203
	s_ashr_i32 s1, s0, 31                                      // 0000000017F0: 86019F00
	v_mov_b32_e32 v1, 0                                        // 0000000017F4: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017F8: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017FC: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001800: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001804: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001808: 82010105
	s_add_u32 s0, s0, s6                                       // 00000000180C: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001810: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 000000001814: DC6A0000 00000001
	s_nop 0                                                    // 00000000181C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001820: BFB60003
	s_endpgm                                                   // 000000001824: BFB00000
