
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_81_4>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000171C: BF8704D9
	s_lshl_b64 s[12:13], s[14:15], 2                           // 000000001720: 848C820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s12                                      // 000000001728: 80060C06
	s_addc_u32 s7, s7, s13                                     // 00000000172C: 82070D07
	s_lshl_b32 s8, s2, 2                                       // 000000001730: 84088202
	s_ashr_i32 s9, s8, 31                                      // 000000001734: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_lshl_b64 s[8:9], s[8:9], 2                               // 00000000173C: 84888208
	s_add_u32 s0, s0, s8                                       // 000000001740: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001744: 82010901
	s_load_b128 s[8:11], s[0:1], null                          // 000000001748: F4080200 F8000000
	s_clause 0x3                                               // 000000001750: BF850003
	s_load_b32 s0, s[6:7], null                                // 000000001754: F4000003 F8000000
	s_load_b32 s1, s[6:7], 0x144                               // 00000000175C: F4000043 F8000144
	s_load_b32 s3, s[6:7], 0x288                               // 000000001764: F40000C3 F8000288
	s_load_b32 s6, s[6:7], 0x3cc                               // 00000000176C: F4000183 F80003CC
	s_waitcnt lgkmcnt(0)                                       // 000000001774: BF89FC07
	v_fma_f32 v0, s0, s8, 0                                    // 000000001778: D6130000 02001000
	s_mul_i32 s0, s2, 0x51                                     // 000000001780: 9600FF02 00000051
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001788: BF8704A1
	v_fmac_f32_e64 v0, s1, s9                                  // 00000000178C: D52B0000 00001201
	s_ashr_i32 s1, s0, 31                                      // 000000001794: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001798: 84808200
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000179C: BF870001
	v_fmac_f32_e64 v0, s3, s10                                 // 0000000017A0: D52B0000 00001403
	s_add_u32 s0, s4, s0                                       // 0000000017A8: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017AC: 82010105
	s_add_u32 s0, s0, s12                                      // 0000000017B0: 80000C00
	s_addc_u32 s1, s1, s13                                     // 0000000017B4: 82010D01
	v_fmac_f32_e64 v0, s6, s11                                 // 0000000017B8: D52B0000 00001606
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017C0: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 0000000017C4: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 0000000017CC: DC6A0000 00000001
	s_nop 0                                                    // 0000000017D4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D8: BFB60003
	s_endpgm                                                   // 0000000017DC: BFB00000
